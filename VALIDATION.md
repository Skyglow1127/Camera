# 本机验证记录

验证日期：2026-10-14（用户时区 Asia/Singapore）。主机为 AMD Ryzen AI 9 HX 370；C++ OpenCV 4.6.0，ONNX Runtime CPU 1.31.0，推理线程 4，Release 构建。

## 速度比较

使用同一份实拍 RGB-D 图像，模型输入 416×416，3 次预热。分割与几何模式各测 30 次，YOLOv4 测 15 次。下表为完整视觉处理耗时，包含预处理、模型推理、后处理和当前画面的深度分析，不包含相机采集、显示和磁盘保存。

| 模式 | 处理耗时中位数 | P95 | 模型推理中位数 |
| --- | ---: | ---: | ---: |
| YOLO26n-seg / ONNX Runtime | 16.21 ms | 18.86 ms | 13.77 ms |
| YOLO11n-seg / ONNX Runtime | 19.90 ms | 21.57 ms | 17.15 ms |
| YOLOv4 / OpenCV DNN | 273.45 ms | 297.50 ms | 271.30 ms |
| 桌面几何模式 | 0.56 ms | 0.61 ms | 无模型 |

默认选择 YOLO26n-seg：这份样本上比原 YOLOv4 更快，同时提供物体掩膜。YOLO11n-seg 留作备选，桌面几何模式用于分开放置的未知类别物体。

这些数字是**单场景性能测试，不是精度评测，也不是任意场景保证**。几何模式处理时间受物体数量、有效深度和连通区域复杂度影响；当前场景并非最终末端相机俯视工位，不能据此声称已测出抓取成功率。

机器可读记录：`output/bench_yolo26/benchmark.json`、`output/bench_yolo11/benchmark.json`、`output/bench_yolov4/benchmark.json`、`output/bench_tabletop/benchmark.json`。对应原始画面保存在 `output/benchmark_fixture/last_session.yml`，便于用户后续运行覆盖默认输出后仍可复现。

## 实机与窗口

- 实际识别到 DaBai DC1；使用彩色 640×480、深度原始 640×400、30 FPS，硬件 D2C 后为彩色图坐标。
- SDK 将设备 Y11/Y12 等深度传输格式解包成 16 位数据，程序检查数据尺寸后读取。
- 检测到设备不支持 SDK `enableFrameSync()`，而 RGB/depth 设备时间戳域不同。采用主机接收时间缓冲近邻配对，时间门限 25 ms，明确标记为静止场景近似配对。
- 默认分割模式连续采集测试中，启动后采集和处理均约 29.9 FPS；窗口模式也已实际启动、显示并正常退出。
- 实时画面独立于推理刷新；匹配结果图显示结果年龄。无机械臂通信，无运动指令。

## 分割解码对照

用同一份 ONNX 模型、同一份图像，与 Ultralytics 官方 Python 后处理独立比对；结果保存在对应 `output/bench_*/reference_validation.json`。

| 模型 | 检查的两个实例的掩膜 IoU | 最大框坐标差 |
| --- | --- | --- |
| YOLO26n-seg | 0.9991、0.9865 | 小于 1 像素 |
| YOLO11n-seg | 0.9987、0.9757 | 小于 1 像素 |

此项证明 C++ 的模型输出解码和掩膜映射与参考实现接近，**不代表物体识别准确率**。置信度差异小于 1e-4。

复现命令：

```bash
.venv-models/bin/python scripts/validate_segmentation.py \
  --model models/yolo26n-seg.onnx \
  --image output/benchmark_fixture/last_session_color.png \
  --cpp-output output/bench_yolo26
```

## 自动测试与标定

`ctest --test-dir build --output-on-failure` 通过；`vision_tests` 包含 29 个断言，覆盖：

- 已知桌面和物体深度的三维定位、物体高度与夹持宽度。
- 几何模式识别不在预训练类别中的分离物体。
- 实例掩膜排除背景；深度观测和抓取有效性分离。
- 零深度、NaN、中心缺失、无桌面、无物体、不匹配尺寸与超出夹爪开口。
- 同类别重叠框抑制、不同类别重叠框保留。
- 非刚体矩阵、反射矩阵、NaN 变换拒绝。
- 18 组独立生成的眼在手上观测恢复已知相机到末端变换，拒绝退化旋转数据。
- 位姿乘法顺序、过期机器人位姿和运动中位姿拒绝。
- RGB-D 保存/重放及毫米到米转换。

手眼标定命令行程序也通过合成观测验证。真实手眼标定仍需真实 D1-T 位姿和标定板图像；合成测试不能替代实物标定。

## 尚未验证的实际效果

- 最终俯视工作距离下的绝对深度误差与重复定位精度，尚无尺量/标定目标真值。
- D1-T 的真实手眼标定、夹爪开口、TCP 偏移、接触高度、运动规划与抓取成功率。
- 复杂遮挡、透明/反光材质，以及自定义物体类别的召回率。

建议下一轮使用同一物体、多个已知位置和多次停稳拍摄，分别记录三维偏差与抖动；随后再接机械臂抓取测试。
