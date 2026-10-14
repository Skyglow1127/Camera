#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <libobsensor/ObSensor.hpp>

#ifndef CAMERA_ASSET_DIR
#define CAMERA_ASSET_DIR "."
#endif

// 读取类别名称
std::vector<std::string> loadClassNames(const std::string& path) {
    std::vector<std::string> classNames;
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) classNames.push_back(line);
    }
    return classNames;
}

int main() {
    std::cout << "正在启动设备连接..." << std::endl;

    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Context ctx;

    auto devList = ctx.queryDeviceList();
    if (devList->deviceCount() == 0) {
        std::cerr << "未发现设备！" << std::endl;
        return -1;
    }

    auto dev = devList->getDevice(0);
    ob::Pipeline pipe(dev);
    std::cout << "设备连接成功，型号：" << dev->getDeviceInfo()->name() << std::endl;

    // 模型目录由 CMake 根据当前项目位置提供，不再绑定用户主目录。
    const std::string assetDir = CAMERA_ASSET_DIR;
    const std::string modelWeights = assetDir + "/yolov4.weights";
    const std::string modelConfig = assetDir + "/yolov4.cfg";
    const std::string classesFile = assetDir + "/coco.names";
    
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "AI 模型加载成功！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "模型加载失败: " << e.what() << std::endl;
        return -1;
    }

    std::vector<std::string> classNames = loadClassNames(classesFile);
    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();

    pipe.start();
    std::cout << "视频流已启动，按 'q' 或 ESC 键退出..." << std::endl;

    cv::namedWindow("Orbbec AI Detection", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Orbbec Depth Viewer", cv::WINDOW_AUTOSIZE);

    while (true) {
        auto frameSet = pipe.waitForFrames(1000);
        if (frameSet == nullptr) continue;

        // ========== 1. 处理彩色图 + AI 检测 ==========
        auto colorFrame = frameSet->colorFrame();
        if (colorFrame != nullptr) {
            cv::Mat colorMat;
            auto format = colorFrame->format();

            if (format == OB_FORMAT_MJPG) {
                cv::Mat rawData(1, colorFrame->dataSize(), CV_8UC1, colorFrame->data());
                colorMat = cv::imdecode(rawData, cv::IMREAD_COLOR);
            } else if (format == OB_FORMAT_RGB) {
                colorMat = cv::Mat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
                cv::cvtColor(colorMat, colorMat, cv::COLOR_RGB2BGR);
            } else if (format == OB_FORMAT_BGR) {
                colorMat = cv::Mat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
            }

            if (!colorMat.empty()) {
                // --- AI 推理 ---
                cv::Mat blob;
                cv::dnn::blobFromImage(colorMat, blob, 1/255.0, cv::Size(416, 416), cv::Scalar(), true, false);
                net.setInput(blob);
                std::vector<cv::Mat> outs;
                net.forward(outs, outNames);

                std::vector<int> classIds;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes;

                for (size_t i = 0; i < outs.size(); ++i) {
                    float* data = (float*)outs[i].data;
                    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                        cv::Point classIdPoint;
                        double confidence;
                        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                        if (confidence > 0.5) {
                            int centerX = (int)(data[0] * colorMat.cols);
                            int centerY = (int)(data[1] * colorMat.rows);
                            int width = (int)(data[2] * colorMat.cols);
                            int height = (int)(data[3] * colorMat.rows);
                            boxes.push_back(cv::Rect(centerX - width/2, centerY - height/2, width, height));
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back((float)confidence);
                        }
                    }
                }

                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

                // --- 获取深度图，用于读取距离 ---
                cv::Mat depthMat;
                auto depthFrame = frameSet->depthFrame();
                if (depthFrame != nullptr) {
                    depthMat = cv::Mat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data()).clone();
                }

                // --- 绘制检测框和距离信息 ---
                for (int idx : indices) {
                    cv::Rect box = boxes[idx];
                    cv::rectangle(colorMat, box, cv::Scalar(0, 255, 0), 2);

                    std::string label = classNames[classIds[idx]] + ": " + cv::format("%.2f", confidences[idx]);

                    // 如果深度图存在，尝试读取框中心点的深度
                    if (!depthMat.empty()) {
                        int centerX = box.x + box.width / 2;
                        int centerY = box.y + box.height / 2;

                        // 【关键】坐标映射：彩色图 640x480，深度图 640x400
                        int depthX = centerX * depthMat.cols / colorMat.cols;
                        int depthY = centerY * depthMat.rows / colorMat.rows;

                        if (depthX >= 0 && depthX < depthMat.cols && depthY >= 0 && depthY < depthMat.rows) {
                            uint16_t depthValue = depthMat.at<uint16_t>(depthY, depthX);
                            if (depthValue > 0) {
                                // 深度单位通常是毫米，除以1000转成米
                                label += " | " + cv::format("%.2fm", depthValue / 1000.0);
                            }
                        }
                    }

                    cv::putText(colorMat, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                }

                cv::imshow("Orbbec AI Detection", colorMat);
            }
        }

        // ========== 2. 处理深度图显示 ==========
        auto depthFrame = frameSet->depthFrame();
        if (depthFrame != nullptr) {
            cv::Mat depthMat(depthFrame->height(), depthFrame->width(), CV_16UC1, depthFrame->data());
            cv::Mat depth8u;
            double minVal, maxVal;
            cv::minMaxLoc(depthMat, &minVal, &maxVal);
            if (maxVal > 0) {
                depthMat.convertTo(depth8u, CV_8U, 255.0 / maxVal);
            } else {
                depthMat.convertTo(depth8u, CV_8U, 1);
            }
            cv::Mat depthColor;
            cv::applyColorMap(depth8u, depthColor, cv::COLORMAP_JET);
            cv::imshow("Orbbec Depth Viewer", depthColor);
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    pipe.stop();
    cv::destroyAllWindows();
    std::cout << "程序结束，设备已断开。" << std::endl;
    return 0;
}
