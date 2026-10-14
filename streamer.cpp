#include <iostream>
#include <opencv2/opencv.hpp>
#include <libobsensor/ObSensor.hpp>

int main() {
    std::cout << "正在启动设备连接..." << std::endl;

    // 1. 初始化 SDK，调低日志等级
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Context ctx;

    // 2. 查询并连接设备
    auto devList = ctx.queryDeviceList();
    if (devList->deviceCount() == 0) {
        std::cerr << "未发现设备！请检查 USB 连接。" << std::endl;
        return -1;
    }

    auto dev = devList->getDevice(0);
    ob::Pipeline pipe(dev);
    std::cout << "设备连接成功，型号：" << dev->getDeviceInfo()->name() << std::endl;

    // 3. 启动视频流
    pipe.start();
    std::cout << "视频流已启动，按 'q' 或 ESC 键退出..." << std::endl;

    cv::namedWindow("Orbbec Color Viewer", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Orbbec Depth Viewer", cv::WINDOW_AUTOSIZE);

    // 4. 主循环：持续获取画面
    while (true) {
        auto frameSet = pipe.waitForFrames(1000);
        if (frameSet == nullptr) {
            std::cerr << "获取画面超时..." << std::endl;
            continue;
        }

        // --- 处理彩色图 ---
        auto colorFrame = frameSet->colorFrame();   // 正确方法名
        if (colorFrame != nullptr) {
            cv::Mat colorMat;
            auto format = colorFrame->format();      // 正确：format() 而非 getFormat()

            if (format == OB_FORMAT_MJPG) {
                // 正确：dataSize() 和 data()
                cv::Mat rawData(1, colorFrame->dataSize(), CV_8UC1, colorFrame->data());
                colorMat = cv::imdecode(rawData, cv::IMREAD_COLOR);
            } else if (format == OB_FORMAT_RGB) {
                // 正确：width() 和 height()
                colorMat = cv::Mat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
                cv::cvtColor(colorMat, colorMat, cv::COLOR_RGB2BGR);
            } else if (format == OB_FORMAT_BGR) {
                colorMat = cv::Mat(colorFrame->height(), colorFrame->width(), CV_8UC3, colorFrame->data());
            }

            if (!colorMat.empty()) {
                cv::imshow("Orbbec Color Viewer", colorMat);
            }
        }

        // --- 处理深度图 ---
        auto depthFrame = frameSet->depthFrame();   // 正确方法名
        if (depthFrame != nullptr) {
            // 正确：height(), width(), data()
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

        // 5. 检测按键，按 'q' 或 ESC 退出
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }

    // 6. 停止设备并释放资源
    std::cout << "正在停止视频流..." << std::endl;
    pipe.stop();
    cv::destroyAllWindows();
    std::cout << "程序结束，设备已断开。" << std::endl;

    return 0;
}