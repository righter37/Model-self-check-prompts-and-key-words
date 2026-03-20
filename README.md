🔍 内容概览
这份提示词包含以下核心模块：
1️⃣ 日志记录规范
环境信息（Python/PyTorch/CUDA/GPU版本）
实验配置（超参数、模型架构）
训练过程（Loss、Accuracy、学习率、显存使用）
日志文件结构（train.log、metrics.json、config.json）
2️⃣ Checkpoint管理规范
多Stage隔离目录（解决覆盖问题）
plain
复制
checkpoints/
├── stage1_pretrain/
│   ├── best/、latest/、periodic/
├── stage2_finetune/
│   └── ...
命名规范：checkpoint_best_epoch_0050_step_00125000_metric_0.9234.pth
完整元数据（epoch、optimizer状态、Git版本、父checkpoint路径）
原子写入防止文件损坏
3️⃣ 实时监控规范
控制台实时输出（Epoch、Step、Loss、LR、ETA、GPU显存）
TensorBoard/wandb集成
异常检测（NaN/爆炸/停滞/梯度异常）
4️⃣ CV/医学图像特有规范
预处理: 窗宽窗位（CT）、重采样、归一化
数据增强: 弹性形变、噪声、亮度对比度
损失函数: Focal Loss、Dice Loss、Combined Loss
评估指标: Dice、IoU、HD95、敏感度、特异度、AUC-ROC
内存管理: 3D Patch采样、混合精度训练
🚀 使用方法
复制主提示词内容给编程助手
填写项目具体信息部分（任务类型、数据类型、模型等）
让助手生成完整代码
使用快速检查表审查生成的代码
