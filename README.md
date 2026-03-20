## 🔍 内容概览

这份规范整合了4位专家的建议，包含以下核心模块：

1. 日志记录规范
2. Checkpoint管理规范
3. 实时监控规范
4. CV/医学图像特有规范

---

## 1️⃣ 日志记录规范

### 1.1 环境信息

| 记录项目 | 重要性 | 说明 |
|---------|--------|------|
| Python版本 | ⭐⭐⭐⭐⭐ | 不同版本可能导致API行为差异 |
| PyTorch版本 | ⭐⭐⭐⭐⭐ | 框架版本影响模型行为和性能 |
| CUDA版本 | ⭐⭐⭐⭐⭐ | 影响GPU计算结果的一致性 |
| GPU型号与数量 | ⭐⭐⭐⭐ | 硬件差异影响训练速度和内存 |
| 操作系统 | ⭐⭐⭐⭐ | 系统调用和环境变量差异 |

### 1.2 实验配置

- **超参数**: 学习率、Batch Size、优化器类型、学习率调度策略
- **模型架构**: 网络结构、参数量、模型大小
- **数据配置**: 数据集划分、预处理流程、数据增强策略

### 1.3 训练过程指标

| 指标类型 | 记录内容 |
|---------|---------|
| 损失值 | Train Loss、Val Loss |
| 准确率 | Accuracy、Top-k Accuracy |
| 学习率 | 当前学习率值 |
| 显存使用 | GPU Memory Allocated/Reserved |
| 时间指标 | 每步耗时、每轮耗时、ETA |

### 1.4 日志文件结构

```
experiments/
├── {experiment_name}_{timestamp}/
│   ├── logs/
│   │   ├── train.log          # 主训练日志
│   │   ├── error.log          # 错误日志
│   │   └── metrics.json       # 结构化指标数据
│   ├── config/
│   │   ├── config.json        # 实验配置
│   │   └── model_architecture.txt  # 模型架构
│   └── metadata/
│       ├── git_info.txt       # Git版本信息
│       └── environment.yaml   # 环境配置
```

---

## 2️⃣ Checkpoint管理规范

### 2.1 多Stage隔离目录（解决覆盖问题）

```
checkpoints/
├── stage1_pretrain/           # Stage 1: 预训练
│   ├── best/                  # 最佳模型
│   ├── latest/                # 最新检查点（用于恢复）
│   └── periodic/              # 定期保存
│
├── stage2_finetune/           # Stage 2: 微调
│   ├── best/
│   ├── latest/
│   └── periodic/
│
└── stage3_distill/            # Stage 3: 蒸馏
    └── ...
```

### 2.2 命名规范

**标准格式:**
```
checkpoint_{type}_epoch_{epoch:04d}_step_{step:08d}_metric_{metric:.4f}.pth
```

**命名示例:**
| 类型 | 文件名示例 |
|-----|-----------|
| 最佳模型 | `checkpoint_best_epoch_0050_step_00125000_metric_0.9234.pth` |
| 定期保存 | `checkpoint_periodic_epoch_0010_step_00025000.pth` |
| 最新检查点 | `checkpoint_latest.pth` |
| 中断恢复点 | `checkpoint_interrupt_epoch_0025_step_00062500.pth` |

### 2.3 完整元数据

每个Checkpoint必须包含:

```python
REQUIRED_METADATA = {
    # 训练状态
    'epoch': int,                    # 当前epoch
    'global_step': int,              # 全局步数
    'model_state_dict': dict,        # 模型参数
    'optimizer_state_dict': dict,    # 优化器状态
    'scheduler_state_dict': dict,    # 学习率调度器状态
    
    # 训练配置
    'config': dict,                  # 训练配置快照
    'stage': str,                    # 当前stage名称
    'stage_version': str,            # stage版本
    
    # 性能指标
    'metrics': {
        'train_loss': float,
        'val_loss': float,
        'val_accuracy': float,
        'best_metric': float,
    },
    
    # 系统信息
    'timestamp': str,                # ISO格式时间戳
    'pytorch_version': str,          # PyTorch版本
    'cuda_version': str,             # CUDA版本
    'python_version': str,           # Python版本
    'hostname': str,                 # 训练机器
    'git_commit': str,               # 代码版本
    
    # 版本控制
    'version': str,                  # checkpoint版本
    'parent_checkpoint': str,        # 父checkpoint路径
    'stage_history': list,           # stage历史
}
```

### 2.4 原子写入防止文件损坏

```python
def atomic_save_checkpoint(state_dict, filepath):
    """
    原子性保存checkpoint
    防止写入过程中断导致文件损坏
    """
    dir_name = os.path.dirname(filepath)
    fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(state_dict, f)
        os.replace(temp_path, filepath)  # 原子重命名
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
```

---

## 3️⃣ 实时监控规范

### 3.1 控制台实时输出

**推荐输出格式:**
```
================================================================================
Epoch [3/100] | Step [1250/5000] | LR: 1.00e-04 | ETA: 02:15:30
--------------------------------------------------------------------------------
Train Loss: 0.5234 | Train Acc: 87.25% | Time: 0.125s/step
Val   Loss: 0.4891 | Val   Acc: 89.12% | Best: 89.45% (Epoch 2)
--------------------------------------------------------------------------------
GPU Memory: 4.2GB/8.0GB (52%) | GPU Util: 85% | Grad Norm: 2.34
================================================================================
```

**核心输出指标:**
| 指标 | 输出频率 | 说明 |
|-----|---------|------|
| Epoch | 每轮 | 当前训练轮次 |
| Step | 每步/每N步 | 当前迭代步数 |
| Loss | 每步 | 当前批次损失值 |
| LR | 每步/每轮 | 当前学习率 |
| ETA | 每轮 | 预计完成时间 |
| GPU显存 | 每轮 | 显存使用情况 |

### 3.2 TensorBoard/wandb集成

**推荐工具对比:**
| 工具 | 推荐指数 | 优点 | 适用场景 |
|-----|---------|------|---------|
| TensorBoard | ⭐⭐⭐⭐⭐ | 免费、与PyTorch深度集成 | 本地开发、中小型项目 |
| Weights & Biases | ⭐⭐⭐⭐⭐ | 云端存储、协作友好 | 团队协作、生产环境 |
| MLflow | ⭐⭐⭐⭐ | 开源、模型版本管理 | MLOps、模型管理 |

**必须记录的内容:**
- Scalar: Loss、Accuracy、Learning Rate
- Histogram: 权重分布、梯度分布
- Image: 预测结果可视化（医学图像必须）

### 3.3 异常检测

| 异常类型 | 检测条件 | 严重程度 | 建议操作 |
|---------|---------|---------|---------|
| Loss NaN/Inf | `loss != loss` 或 `loss == inf` | 🔴 严重 | 立即停止，检查数据/学习率 |
| Loss爆炸 | loss > 历史均值 × 10 | 🔴 严重 | 降低学习率，检查梯度裁剪 |
| Loss停滞 | 连续50步loss变化 < 1e-6 | 🟡 警告 | 调整学习率，检查数据 |
| 梯度爆炸 | grad_norm > 100 | 🔴 严重 | 启用梯度裁剪 |
| 梯度消失 | grad_norm < 1e-7 | 🟡 警告 | 检查激活函数、初始化 |
| 过拟合 | val_loss上升，train_loss下降 | 🟡 警告 | 增加正则化、早停 |
| GPU OOM | CUDA out of memory | 🔴 严重 | 减小batch size |

---

## 4️⃣ CV/医学图像特有规范

### 4.1 预处理

**医学图像特有:**
| 预处理步骤 | 适用场景 | 说明 |
|-----------|---------|------|
| 窗宽窗位 | CT图像 | 调整显示范围，突出病灶区域 |
| 重采样 | 3D医学图像 | 统一体素间距 |
| 归一化 | 所有图像 | z-score或min-max归一化 |

### 4.2 数据增强

**通用增强:**
- 几何变换: 翻转、旋转、缩放、裁剪
- 强度变换: 亮度、对比度、Gamma校正

**医学图像特有:**
| 增强类型 | 说明 |
|---------|------|
| 弹性形变 | 模拟组织变形 |
| 高斯噪声 | 模拟设备噪声 |
| 随机亮度对比度 | 适应不同设备参数 |

### 4.3 损失函数

| 损失函数 | 适用场景 | 特点 |
|---------|---------|------|
| Focal Loss | 类别不平衡 | 关注难分类样本 |
| Dice Loss | 医学图像分割 | 优化重叠区域 |
| Combined Loss | 综合场景 | Dice + CE组合 |

### 4.4 评估指标

**分割任务:**
| 指标 | 说明 | 计算公式 |
|-----|------|---------|
| Dice | Dice系数 | 2\|X∩Y\| / (\|X\|+\|Y\|) |
| IoU | 交并比 | \|X∩Y\| / \|X∪Y\| |
| HD95 | 95% Hausdorff距离 | 边界距离评估 |
| 敏感度 | Sensitivity/Recall | TP / (TP + FN) |
| 特异度 | Specificity | TN / (TN + FP) |

**分类任务:**
- Accuracy、Precision、Recall、F1-Score
- AUC-ROC、AUC-PR

### 4.5 内存管理

| 优化策略 | 适用场景 | 说明 |
|---------|---------|------|
| 3D Patch采样 | 3D医学图像 | 分块处理大体积数据 |
| 混合精度训练 | 所有GPU训练 | AMP减少显存占用 |
| DataLoader优化 | 所有训练 | pin_memory、persistent_workers |

---

## 🚀 使用方法

### 步骤1: 复制主提示词
将 `dl_training_prompt.md` 的内容提供给编程助手。

### 步骤2: 填写项目具体信息
```markdown
【任务类型】: {classification/segmentation/detection}
【数据类型】: {2D图像/3D医学图像/多模态医学图像}
【数据路径】: {data_directory}
【模型类型】: {ResNet/U-Net/Transformer/自定义}
【训练阶段】: {单阶段/多阶段预训练+微调}
【类别数量】: {num_classes}
【图像尺寸】: {image_size}
【Batch Size】: {batch_size}
【学习率】: {learning_rate}
【训练轮数】: {epochs}
```

### 步骤3: 让助手生成完整代码
助手将生成:
1. 完整的配置文件
2. 完整的训练脚本（包含上述所有规范）
3. 数据加载器（包含预处理和数据增强）
4. 模型定义
5. 损失函数和评估指标
6. 启动脚本

### 步骤4: 使用快速检查表审查代码

```markdown
### 日志记录检查
- [ ] 环境信息是否记录？
- [ ] 超参数是否保存到JSON？
- [ ] 每个epoch的指标是否记录？
- [ ] 日志文件是否按结构存放？

### Checkpoint检查
- [ ] 多stage是否有独立目录？
- [ ] checkpoint命名是否符合规范？
- [ ] 是否包含完整元数据？
- [ ] 是否实现原子写入？
- [ ] 是否可以断点续训？

### 实时监控检查
- [ ] 控制台是否有实时输出？
- [ ] 是否集成TensorBoard/wandb？
- [ ] 是否有异常检测机制？
- [ ] 是否记录GPU显存使用？

### CV/医学图像特有检查
- [ ] 预处理参数是否正确？
- [ ] 数据增强是否合适？
- [ ] 损失函数是否适合任务？
- [ ] 评估指标是否完整？
- [ ] 3D数据是否实现Patch采样？
```

---

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| Checkpoint被覆盖 | 使用多stage目录隔离 + 原子写入 |
| 无法查看历史记录 | 必须记录metrics.json + TensorBoard |
| 无法断点续训 | checkpoint必须包含optimizer/scheduler状态 |
| 训练异常未被发现 | 实现异常检测机制（NaN/爆炸/停滞） |
| 复现不了结果 | 设置随机种子 + 记录环境信息 |
| GPU OOM | 使用混合精度 + Patch采样 + 梯度累积 |

---
