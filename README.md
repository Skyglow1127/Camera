# D1-T 末端相机抓取视觉程序

当前版本默认使用 **YOLO26n-seg + ONNX Runtime CPU + RGB-D 几何定位**。适用于机械臂停稳观察、桌面上分开放置的物体。已在本机连接的 **Orbbec DaBai DC1** 上运行验证。

程序完成视觉检测、物体掩膜、深度测量、几何抓取候选、数据保存和离线手眼标定。**没有接入 D1-T 的通信 SDK，也不会发送机械臂运动命令。** 基座坐标需要真实手眼标定和对应机械臂位姿；夹爪 TCP、接触深度、碰撞检查和运动规划仍需实机对接。

## 启动

```bash
/home/aup/Camera/run.sh
```

脚本会构建 Release 版本，再打开两个窗口：

- `Live - latest frame`：持续刷新最新彩色画面。
- `Detection - matched RGB-D`：显示检测时刻的画面、物体轮廓、深度和候选夹持方向。旧检测结果不叠加到新画面。

预训练权重和 ONNX Runtime 已保存在项目内，正常启动不需要 Python、不需要联网。

| 按键 | 操作 |
| --- | --- |
| `空格` | 暂停/继续连续检测，实时预览继续刷新 |
| `d` | 切换到单次检测，处理最新配对帧并停留在结果上 |
| `Tab` | 在当前检测结果中选择目标，新结果到来会取消旧选择 |
| `s` | 保存当前结果、选择索引、彩色图、对齐深度、相机参数和标注图 |
| `c` | 保存当前 RGB-D 和相机参数，供手眼标定或离线复现 |
| `v` | 显示/隐藏深度对齐叠加图，检查深度边缘是否对应物体 |
| `q` / `Esc` | 退出并释放相机 |

建议第一次使用：相机俯视桌面，桌面在物体周围充分可见；机械臂停稳，按 `d`，查看轮廓与深度，按 `Tab` 选择，再按 `s` 保存。保存的旧结果仍可诊断，但 JSON 会标记 `fresh_at_write=0`。

## 三种算法模式

```bash
# 默认：类别 + 实例分割 + 深度定位
./build/detect --mode segment

# 无类别限制：桌面平面 + 高于桌面的独立深度区域
./build/detect --mode tabletop

# 对照：原 YOLOv4 检测框 + 修正后的深度定位
./build/detect --mode yolo

# 替换为已附带的 YOLO11n-seg
./build/detect --mode segment --segment-model models/yolo11n-seg.onnx
```

`segment` 默认使用 YOLO26n-seg，输入 416×416。处理采用等比例补边，掩膜恢复至原图；支持 YOLO26 的端到端输出及 YOLO11 的传统输出。传统输出按类别 NMS；不重复相乘 objectness。

`tabletop` 适合普通盒子、零件等 COCO 模型未训练的物体，不需要训练权重。它假设占主要面积的可见平面就是桌面，并且相机大致俯视。不能用它可靠处理堆叠物体、相接触物体、严重缺失深度的材质或任意朝向的场景。相机正对显示器/墙壁时，也可能把它们识别成参考平面，因此不能把“table OK”理解为自动理解了桌面。

分割模型的 80 个类别并不包括任意零件和通用纸盒；模型分类限制可以用 `tabletop` 绕开，后续也可用实际物体样本微调分割模型。当前默认显示所有类别，但 `person` 不生成有效抓取候选。

## 深度和候选的含义

1. 选择相机支持的硬件 D2C；没有硬件支持时尝试软件 D2C。
2. 深度乘 SDK 的 `getValueScale()` 后转成米，要求深度、彩色图和内参尺寸一致。
3. 支持同步接口的设备使用设备时间戳门限。DaBai DC1 不支持此接口，且彩色与深度设备时间戳不在同一时钟域，因此使用最多 8 帧彩色缓冲、按主机接收时间就近配对。
4. DaBai 的这种配对是**静止场景近似配对，不是曝光同步**。机械臂运动时只能预览；抓取定位需停稳后重新采集。不能把它直接用于运动中的视觉伺服。
5. 对物体内部有效深度做稳健测量；屏幕 `Z` 是沿相机光轴的深度，不是到相机原点的欧氏距离。
6. 拟合平面，剔除桌面/背景，提取物体区域，估计表面点和夹持方向。没有可信平面时仍可显示独立深度测量，但不输出有效抓取姿态。
7. 夹持候选 `x` 轴表示闭合方向，`z` 轴表示朝向桌面的接近方向。圆形等对称物体的方向不唯一，`orientation_anisotropy` 接近 1 时不要把方向当成稳定姿态。

候选开口来自可见表面在桌面平面内的宽度估计，并加两侧余量，不等于保证夹得住。遮挡、软物体、透明瓶体/反光金属、物体侧面不可见都可能使几何估计失真。缺失深度不通过补洞伪装成真实测量。

## 配置

修改 `grasp_config.yml`；其中的相对文件路径以配置文件目录为基准。

| 参数 | 用途 |
| --- | --- |
| `detection_mode` | `segment`、`tabletop`、`yolo` |
| `classes` | 只保留指定类别，例如 `[ "bottle", "cup" ]`；空数组保留全部 |
| `excluded_grasp_classes` | 显示识别结果但不生成有效抓取候选的类别 |
| `cpu_threads` | CPU 推理线程数，默认 4，避免推理占满所有线程影响显示 |
| `confidence` | 默认 0.35，调低可增加候选但也可能增加误检 |
| `min_depth_m` / `max_depth_m` | 有效深度范围，按实际工作距离调整 |
| `plane_threshold_m` | 平面拟合的距离容差 |
| `min_object_height_m` | 排除桌面的最小物体高度，薄物体需适当减小 |
| `max_gripper_opening_m` | **当前 0.080 只是占位值，不是 D1-T 夹爪规格，必须按实际夹爪修改** |
| `finger_margin_m` | 估计开口两侧各增加的余量 |
| `max_rgbd_skew_ms` | 时间配对门限；放宽不等于获得真实同步 |
| `handeye_file` / `robot_pose_file` | 未配置时只输出相机坐标 |

分割模式的 `input_size` 必须与固定尺寸 ONNX 模型一致；不能只改 YAML 就切换分辨率。重新导出 320 或 640 输入时，还应重新做速度和识别效果比较。

## 输出与复现

输出目录默认 `output/`。每次处理后原子更新 `latest.json`，退出时保存 `last_session.yml`、彩色图、毫米深度 PNG、标注图和深度叠加图。

关键字段：

- `rgbd_pair_accepted`：当前时间门限下配对是否通过；结合 `rgbd_pairing` 理解，不能当作硬同步标志。
- `depth_measurement_valid`、`surface_camera_m`：是否获得物体表面的有效三维测量，相机轴为右、下、前。
- `valid`：仅表示几何候选通过本版检查，不表示可以执行机械臂运动。
- `T_camera_surface`：仅对有效候选输出，相机坐标下的表面候选坐标系。
- `T_base_surface`：仅在手眼标定已验证、机械臂静止且位姿时间匹配时输出。
- `motion_command_ready`：本版恒为 0，因为尚未完成 TCP 接触偏移、碰撞/可达性与运动控制。
- `fresh_at_write` 和 `host_received_ms`：检查数据新鲜程度。文件退出后仍保留，使用者不能把存在的文件视为持续有效的新结果。
- 矩阵使用 OpenCV JSON 形式 `{ rows, cols, dt, data }`，`data` 为行优先。

```bash
# 重放一次保存的 RGB-D
./build/detect --replay output/last_session.yml --output output/replay

# 无窗口相机测试，十秒后退出
./build/detect --headless --seconds 10

# 查看设备及可用分辨率
./build/detect --probe

# 相同画面的速度比较：3 次预热，30 次测量
./build/detect --replay output/last_session.yml --benchmark 30 --output output/benchmark
```

快照深度 PNG 使用毫米量化；原始实时数据以 SDK 比例换算后的浮点米处理。重放会重新读取采集时间信息，不会给旧画面编造新的机械臂位姿。

## D1-T 手眼标定与位姿接口

位置转换约定：`T_base_surface = T_base_end × T_end_camera × T_camera_surface`。所有平移单位为米。`end` 必须是同一个固定的机械臂末端坐标系，不能把法兰坐标和 TCP 坐标混用。

1. 把标定板固定在桌面，不随机械臂移动。
2. 修改 `calibration/samples.example.yml` 中内角点行列数和真实方格边长，复制成自己的样本文件。
3. 在多个静止末端姿态按 `c` 保存图像，同时从 D1-T 控制接口读取该姿态的 `T_base_end`。至少 8 组，建议 15–25 组，包含绕不同轴的旋转。
4. 在 `samples` 中添加 `snapshot` 路径和对应 `T_base_end`，或者直接提供 `T_camera_target`。
5. 运行：

```bash
./build/handeye_calibrate calibration/samples.yml calibration/handeye.yml
```

标定器会拒绝旋转退化的数据，输出固定标定板位置/旋转的一致性 RMS；图片输入还会检查角点重投影误差。输出默认 `verified: 0`。用未参与标定的姿态和已知位置目标验证后，才将真实文件设为 `verified: 1`；低训练残差不代表实际抓取坐标准确。

`calibration/robot_pose.example.yml` 定义了后续 D1-T 位姿适配器的输入：

- 正确的 `T_base_end`、米制平移。
- `stationary: 1` 必须对应实际停稳采集。
- `host_received_ms` 与相机采集的 SDK 主机时间域对应；本版只接受静止时刻附近的位姿。
- 适配器应先写临时文件，再原子替换目标文件，避免读取半写入文件。

身份矩阵模板、时间戳 0 或未验证的标定不会被当作真实基座定位。项目没有猜测 D1-T 的网络协议；后续只需依据真实控制 SDK 对接此位姿接口及运动层。

## 验证与模型来源

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

自动测试覆盖桌面/物体分离、深度缺失、实例掩膜、不同类别 NMS、夹爪宽度约束、已知手眼变换恢复、退化标定和过期机械臂位姿拒绝。实机和模型对照结果见 `VALIDATION.md`。

默认模型来自 [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)，备选为 [YOLO11](https://docs.ultralytics.com/models/yolo11/)。模型及其使用遵循 [Ultralytics 许可](https://www.ultralytics.com/license)，运行库保留了 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 自带许可。模型源地址、导出版本和 SHA256 见 `models/*.json`。

模型准备环境独立保存在 `.venv-models/`，约 1.6 GB，仅用于导出/参考验证，运行 C++ 程序不依赖它。需要重新准备依赖时运行 `scripts/setup_models.sh`；导出代码在 `scripts/prepare_models.py`，安装版本记录在 `models/export-requirements.txt`。

原始检测程序已保留为 `detect_legacy.cpp`，原始 YOLOv4 权重和 `streamer.cpp` 保留。
