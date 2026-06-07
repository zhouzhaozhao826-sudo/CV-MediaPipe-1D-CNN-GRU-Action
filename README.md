# 基于计算机视觉的深蹲动作规范性评估系统

前情提要：
本项目是作者本人亲手制作，不是二次制作。本项目已跑通全流程，可作为入门小项目或者学科作业。改进方案与工作流程是项目说明里最重要的文件务必仔细阅读。数据集是亲自或者朋友拍摄所以不方便公开，上传文件中的export-round3是我最新一轮的数据集训练出来的成果里面有最新的权重可以直接使用，也可以自行获取数据集按照改进方案与工作流程进行工作


## 1. 项目背景

在体能训练和大众健身领域，动作不规范是导致训练效果不佳甚至运动损伤的主要原因。传统的动作评估依赖专业教练肉眼观察，存在主观性强、难以量化、无法大规模普及等问题。

本项目利用 **计算机视觉 + 深度学习** 技术，构建了一套深蹲动作规范性自动评估系统。系统通过摄像头或视频提取人体姿态关键点，计算生物力学特征，对深蹲动作进行自动识别、错误分类和矫正建议输出。

应用场景包括：居家健身智能指导、青少年体测辅助、康复训练监测、以及作为论文实验平台支撑 **ICEICT 国际会议** 投稿。

## 2. 系统目标

- 基于 MediaPipe Pose 提取 33 个人体 3D 关键点，转换为 15 维尺度不变的几何特征
- 实现对深蹲动作的四分类：**标准深蹲 / 下蹲深度不足 / 膝内扣 / 躯干过度前倾**
- 构建三条技术路线并完成对比实验，支撑论文：
  - **规则法**：生物力学阈值 + 状态机，实时判定，可解释性强
  - **深度法**：1D-CNN + GRU 端到端分类，从标注数据中自动学习时空特征
  - **融合法**：规则法 + 深度法加权融合，兼顾准确率与可解释性

## 3. 上传文件说明

```
try_action/
├── README.md                         ← 本文件
├── requirements.txt                  ← Python 依赖清单
│
├── app.py                            ← CLI 入口：摄像头实时规则法 / 视频特征提取+切片
├── run_workflow.py                   ← 主工作流入口：一键完成 分析→标注→建数据集→训练→评估→推理
├── compare.py                        ← 三方法对比评估：准确率 + 混淆矩阵 + JSON 报告
├── export_segments.py                ← 可视化工具：导出每个切片的视频和关键帧
├── _inspect_dataset.py               ← 调试工具：检查 .npz 数据集结构与标签分布
│
├── pose_action/                      ← 核心算法包
│   ├── __init__.py                   ← 包入口，统一导出
│   ├── config.py                     ← 全局配置：路径、摄像头参数、置信度阈值
│   ├── landmarks.py                  ← MediaPipe Pose 封装：BGR帧 → 33个关键点 3D 坐标
│   ├── features.py                   ← 特征工程：33关键点 → 15维尺度不变特征（角度、距离、比值）
│   ├── segment.py                    ← 动作切片：峰值检测 + 8条质量过滤，自动切割视频中的深蹲段
│   ├── rules.py                      ← 规则法核心：膝角>95°/躯干>35°/膝间距<髋宽×0.85 阈值判定
│   ├── repetition.py                 ← 深蹲计数器：EMA 平滑 + 状态机（站立→下降→底部→站起）
│   ├── model.py                      ← 1D-CNN + GRU 网络 (PyTorch)，输入(50帧,15维)→输出 4 类
│   ├── training.py                   ← 训练/评估：早停、学习率衰减、类别权重、L2 正则化
│   ├── dataset.py                    ← 数据集构建：切片→.npz、标签管理、规则法自动预标注
│   └── pipeline.py                   ← 总控管线：特征提取→切片→判定→推理→融合→标注视频
│
├── datasets/                         ← 多轮数据与产出
│   ├── round1/                       ← 第一轮（早期实验，含 segment 可视化视频）
│   ├── round2/                       ← 第二轮（四类标签初步实验）
│   └── round3/                       ← 第三轮（当前轮次）
│       ├── videos_round3/            ← 原始视频（5个，每类/每个角度各一）
│       │   ├── standard/
│       │   │   ├── r3_standard_C.mp4 ← 标准深蹲·侧面90°
│       │   │   └── r3_standard_Z.mp4 ← 标准深蹲·正面
│       │   ├── depth_insufficient/
│       │   │   └── r3_depth_C.mp4    ← 深度不足·侧面90°
│       │   ├── knee_valgus/
│       │   │   └── r3_valgus_Z.mp4   ← 膝内扣·正面
│       │   └── torso_lean/
│       │       └── r3_lean_C.mp4     ← 躯干前倾·侧面90°
│       └── exports_round3/           ← 输出结果（特征CSV、切片结果、标签、数据集.npz、合并数据集）
│
├── 改进方案与工作流程.md              ← 核心设计文档：三条技术路线、15维特征推导、切片算法、操作手册
├── MediaPipe_Comprehensive_Guide.md   ← MediaPipe 技术百科：架构、原理、对比、应用
```

### 各文件作用速查

| 文件 | 一句话作用 |
|:---|:---|
| `app.py` | 摄像头实时规则法 / 视频特征提取+切片，双模式 CLI |
| `run_workflow.py` | **主要入口**：改配置开关，一键完成全流程，支持长视频切片和短视频批量两种模式 |
| `compare.py` | 规则法 vs 深度法 vs 融合法统计对比，输出准确率+混淆矩阵+JSON |
| `export_segments.py` | 把每个动作段导出为小视频+底部/起止关键帧 |
| `_inspect_dataset.py` | 快速检查 .npz 数据集结构、shape、标签分布 |
| `config.py` | 全局路径与参数默认值 |
| `landmarks.py` | BGR帧 → MediaPipe Pose 推理 → 33个3D关键点坐标 |
| `features.py` | 33关键点 → 15维几何特征（7角度 + 3高度 + 3距离 + 2比值） |
| `segment.py` | 髋部高度峰值检测 → 自动切割完整深蹲段（含8条质量过滤） |
| `rules.py` | 规则法核心：bottom帧阈值判定，输出错误类型+矫正建议+置信度 |
| `repetition.py` | EMA平滑 + 状态机，实时追踪站立→下降→底部→站起循环 |
| `model.py` | 1D-CNN + GRU 网络（PyTorch），(50,15) → 4类 logits |
| `training.py` | 训练循环 + 早停 + LR衰减 + 类别权重 + L2正则 |
| `dataset.py` | 切片→.npz训练集；标签模板/录入/汇总；规则法自动预标注 |
| `pipeline.py` | 总控管线：特征提取→切片→规则判定→DL推理→融合→标注视频 |

## 4. 技术架构概览

### 4.1 三条技术路线

```
输入视频/摄像头
        │
        ▼
  MediaPipe Pose (33关键点 3D坐标)
        │
        ▼
  15维尺度不变特征 (角度/距离/比值)
        │
        ├─→ 规则法 (rules.py):         生物力学阈值 + 状态机 → 实时反馈
        │
        ├─→ 深度法 (model.py):         1D-CNN+GRU 端到端分类 → 离线视频分析
        │
        └─→ 融合法 (pipeline.py):      规则法×α + 深度法×(1-α) 加权融合
```

### 4.2 四类动作标签

| 标签 | 名称 | 判定标准 |
|:---|:---|:---|
| 0 | `standard` | 膝角≤95°、膝间距≥髋宽×0.85、躯干倾角≤35° |
| 1 | `depth_insufficient` | 底部膝角 > 95° |
| 2 | `knee_valgus` | 底部膝间距 < 髋宽 × 0.85 |
| 3 | `torso_lean` | 底部躯干倾角 > 35° |

## 5. 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 摄像头实时规则法

```bash
python app.py --source camera
```

### 视频特征提取 + 切片

```bash
python app.py --source video --video-path datasets/round3/videos_round3/standard/r3_standard_C.mp4
```

### 一键全流程（推荐）

编辑 `run_workflow.py` 顶部的 `VIDEO_SOURCE` 和开关，然后：

```bash
python run_workflow.py
```

支持的开关：`RUN_VIDEO_ANALYSIS` / `AUTO_LABEL` / `BUILD_DATASET` / `RUN_TRAINING` / `RUN_EVALUATION` / `RUN_MODEL_INFERENCE`

### 三方法对比评估

```bash
python compare.py \
    --video-path datasets/round3/videos_round3/test.mp4 \
    --checkpoint-path path/to/best_cnn_gru_model.pth \
    --labels-path path/to/labels.csv \
    --output-json compare_report.json
```

## 6. 注意事项

- 训练前至少需要 4 个已标注动作段，且不少于 2 个类别每类至少 1 个样本
- 切片算法会自动过滤过短、幅度不足、起止不稳定等无效段
- 规则法自动预标注（`AUTO_LABEL=True`）可大幅减少人工标注量，但需人工复核边界情况
- 短视频批量模式（`TRAIN_DATA_MODE="short_video_batch"`）无需切片、无需标注，目录名即标签






最终结果示例图


<img width="1344" height="1279" alt="943bc03e040e6346fc6d6e48d9c4c7aa" src="https://github.com/user-attachments/assets/9068561d-4fab-4662-ac90-b504ca4c0d74" />



<img width="890" height="501" alt="image" src="https://github.com/user-attachments/assets/b6779ba4-51cc-4130-aaa1-96e1f324b715" />


