# try_action

第一版工程骨架聚焦于“深蹲动作规范性评估”的最小可运行链路：

- MediaPipe Pose 提取 33 个姿态关键点
- 计算膝角、髋角、躯干倾角等基础特征
- 基于髋部高度进行简易动作计数
- 实时显示骨架与关键指标
- 将逐帧特征导出为 CSV，供后续 1D-CNN + GRU 训练使用

## 文件分层

日常只需要记住下面 4 个入口：

- `run_workflow.py`：**统一工作流入口**，配置开关后一键完成「分析→标注→建数据集→训练→评估→推理」全流程，支持长视频切片和短视频批量两种模式
- `app.py`：实时摄像头或单视频快速预览，适合演示和快速验证
- `compare.py`：三方法对比评估（规则法 vs 深度法 vs 融合法），输出准确率 + 混淆矩阵 + JSON 报告
- `export_segments.py`：导出每个 segment 的小视频和关键帧，方便人工复核
- `_inspect_dataset.py`：快速检查 .npz 数据集结构和标签分布

核心算法都在 `pose_action/` 里，按功能分 5 组：

- `landmarks.py` + `features.py`：姿态点提取与角度/距离特征构建
- `repetition.py` + `rules.py` + `segment.py`：计数、规则判断、动作切片
- `dataset.py` + `model.py` + `training.py`：样本生成+标签管理、模型定义、训练评估
- `config.py` + `pipeline.py`：项目配置、CSV 导出、视频主流程 + 融合推理 + 标注视频生成
- `compare.py`（顶层）：规则法 vs 深度法 vs 融合法的统计对比与报告导出

目录建议固定为：

- `datasets/round1/`：第一轮数据与产物
- `datasets/round2/`：第二轮数据与产物（历史）
- `datasets/round3/videos_round3/`：第三轮原始视频（当前使用）
- `datasets/round3/exports_round3/`：第三轮导出结果、标签、数据集和训练结果

## 运行方式

先安装依赖：

```bash
pip install -r try_action/requirements.txt
```

使用摄像头运行：

```bash
python try_action/app.py --source camera
```

处理本地视频：

```bash
python try_action/app.py --source video --video-path path/to/video.mp4
```

导出每个 segment 的小视频和关键帧：

```bash
python try_action/export_segments.py --video-path try_action/videos/深蹲.mp4 --segments-path try_action/exports/深蹲_features_segments.csv --output-dir try_action/exports/segment_views --export-video --export-frames
```

如果只想先看一部分 segment：

```bash
python try_action/export_segments.py --video-path try_action/videos/深蹲.mp4 --segments-path try_action/exports/深蹲_features_segments.csv --output-dir try_action/exports/segment_views --export-frames --start-id 1 --end-id 10
```

三方法对比评估：

```bash
# 基础对比（无真实标签时仅输出三方法判定对比）
python try_action/compare.py \
    --video-path path/to/video.mp4 \
    --checkpoint-path path/to/best_cnn_gru_model.pth

# 带真实标签（计算准确率+混淆矩阵）
python try_action/compare.py \
    --video-path path/to/video.mp4 \
    --checkpoint-path path/to/best_cnn_gru_model.pth \
    --labels-path path/to/labels.csv \
    --output-json compare_report.json
```

完整工作流（推荐）：直接编辑 `run_workflow.py` 顶部的配置区和开关，然后：

```bash
python try_action/run_workflow.py
```

注意：

- 训练脚本会自动过滤 `label = -1` 的未标注样本
- 当前至少建议有 4 个已标注动作段再启动训练
- 当前至少建议有 2 个类别且每类不只 1 个已标注样本
- 输出目录中会生成最佳模型、训练历史和训练指标 JSON
- 导出 segment 片段后，优先根据 `bottom` 关键帧判断主错误类型，再填写 `深蹲_labels.csv`
- 最新切片算法会自动过滤明显无效的 segment，包括过短、幅度不足、起止站立不稳定和膝角变化不足的片段

## 当前版本的定位

这个版本不是最终论文系统，而是论文主线的工程起点。它优先解决 4 件事：

- 跑通 MediaPipe Pose
- 得到稳定的逐帧结构化特征
- 切入深蹲单动作场景
- 为后续数据标注、动作切片、1D-CNN+GRU 训练做好输入准备

## 已完成 & 可扩展方向

### 已实现

- 深蹲四分类错误标签体系（standard / depth_insufficient / knee_valgus / torso_lean）
- 动作周期自动切片（峰值检测 + 8条质量过滤规则）
- 规则法自动预标注（`AUTO_LABEL` 开关，大幅减少人工标注量）
- 短视频批量处理模式（`TRAIN_DATA_MODE = "short_video_batch"`，无需切片无需标注）
- 训练后推理与三方法对比评估（`compare.py`，准确率 + 混淆矩阵 + JSON 报告）
- 融合法标注视频生成（三行并列显示规则法/深度法/融合法判定）
- 训练优化：Early Stopping + LR Scheduler + Weight Decay + Class Weights

### 可扩展方向

- 增加 YOLOv8-Pose 对比实验
- 扩展到更多动作类型（硬拉、卧推等）
- 增加更多数据增强策略缓解过拟合
