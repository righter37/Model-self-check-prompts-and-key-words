# 深度学习模型训练规范检查提示词

## 提示词使用说明

将此提示词提供给编程助手，让它根据你的具体项目内容生成符合规范的代码。

---

## 主提示词

```
你是一位深度学习训练代码专家。请为我生成/检查PyTorch训练代码，确保代码符合以下所有规范要求。

【项目领域】: 计算机视觉(CV) / 医学图像分析

=== 必须实现的检查清单 ===

## 一、日志记录规范 (必须实现)

### 1.1 环境信息记录
- [ ] 记录Python版本、PyTorch版本、CUDA版本
- [ ] 记录GPU型号与数量、显存大小
- [ ] 记录操作系统信息
- [ ] 记录随机种子值

### 1.2 实验配置记录
- [ ] 记录所有超参数（学习率、batch size、优化器、调度器等）
- [ ] 记录模型架构信息（参数量、模型大小）
- [ ] 记录数据预处理流程
- [ ] 记录数据集划分信息

### 1.3 训练过程记录
- [ ] 每个epoch记录训练/验证损失
- [ ] 每个epoch记录评估指标（Accuracy、Dice、IoU等）
- [ ] 记录学习率变化
- [ ] 记录GPU显存使用情况
- [ ] 记录每个epoch耗时

### 1.4 日志文件结构
```
experiments/
├── {experiment_name}_{timestamp}/
│   ├── logs/
│   │   ├── train.log          # 主训练日志
│   │   ├── error.log          # 错误日志
│   │   └── metrics.json       # 结构化指标数据
│   ├── config/
│   │   ├── config.json        # 配置文件
│   │   └── model_arch.txt     # 模型架构
│   └── metadata/
│       ├── git_info.txt       # Git版本信息
│       └── environment.yaml   # 环境配置
```

## 二、Checkpoint管理规范 (必须实现)

### 2.1 多Stage隔离 (解决覆盖问题)
- [ ] 不同stage的checkpoint必须存放在不同目录
```
checkpoints/
├── stage1_pretrain/
│   ├── best/                  # 最佳模型
│   ├── latest/                # 最新检查点
│   └── periodic/              # 定期保存
├── stage2_finetune/
│   ├── best/
│   ├── latest/
│   └── periodic/
└── stage3_distill/
    └── ...
```

### 2.2 Checkpoint命名规范
- [ ] 文件名格式: `checkpoint_{type}_epoch_{epoch:04d}_step_{step:08d}_metric_{metric:.4f}.pth`
- [ ] 类型包括: best / periodic / latest / interrupt

### 2.3 Checkpoint必须包含的元数据
- [ ] epoch、global_step
- [ ] model_state_dict、optimizer_state_dict、scheduler_state_dict
- [ ] 训练指标（train_loss、val_loss、val_accuracy等）
- [ ] 配置快照
- [ ] stage名称和版本
- [ ] 时间戳、PyTorch版本、CUDA版本
- [ ] Git commit hash
- [ ] 父checkpoint路径（用于追溯）

### 2.4 防止覆盖策略
- [ ] 使用原子写入（先写临时文件，再重命名）
- [ ] 保存最佳模型时保留历史最佳（保留最近N个）
- [ ] 定期清理旧的periodic checkpoint

## 三、实时训练监控规范 (必须实现)

### 3.1 控制台实时输出
- [ ] 每N步输出: Epoch、Step、Loss、LR、ETA
- [ ] 每个epoch输出: Train Loss、Val Loss、Val Acc、Best Metric
- [ ] 输出格式示例:
```
================================================================================
Epoch [3/100] | Step [1250/5000] | LR: 1.00e-04 | ETA: 02:15:30
--------------------------------------------------------------------------------
Train Loss: 0.5234 | Train Acc: 87.25% | Time: 0.125s/step
Val   Loss: 0.4891 | Val   Acc: 89.12% | Best: 89.45% (Epoch 2)
--------------------------------------------------------------------------------
GPU Memory: 4.2GB/8.0GB (52%) | GPU Util: 85%
================================================================================
```

### 3.2 可视化工具集成
- [ ] 集成TensorBoard或wandb
- [ ] 记录scalar: loss、accuracy、learning rate
- [ ] 记录histogram: 权重分布、梯度分布
- [ ] 记录image: 预测结果可视化（医学图像必须）

### 3.3 异常检测
- [ ] 检测Loss NaN/Inf
- [ ] 检测Loss爆炸（>历史均值×10）
- [ ] 检测Loss停滞（连续50步变化<1e-6）
- [ ] 检测梯度爆炸（grad_norm > 100）
- [ ] 检测梯度消失（grad_norm < 1e-7）

## 四、CV/医学图像特有规范 (必须实现)

### 4.1 数据预处理
- [ ] 医学图像: 实现窗宽窗位调整（CT）
- [ ] 医学图像: 实现重采样到统一体素间距
- [ ] 实现归一化（z-score或min-max）
- [ ] 记录预处理参数

### 4.2 数据增强
- [ ] 几何变换: 翻转、旋转、缩放、裁剪
- [ ] 医学图像特有: 弹性形变
- [ ] 强度变换: 亮度、对比度、Gamma
- [ ] 噪声: 高斯噪声（模拟设备噪声）

### 4.3 损失函数选择
- [ ] 类别不平衡: 使用Focal Loss或Weighted CE
- [ ] 分割任务: 使用Dice Loss或Dice+CE组合
- [ ] 医学图像: 考虑边界感知损失

### 4.4 评估指标
- [ ] 分割任务必须: Dice系数、IoU、HD95
- [ ] 分类任务必须: Accuracy、Precision、Recall、F1、AUC-ROC
- [ ] 医学图像特有: 敏感度(Sensitivity)、特异度(Specificity)

### 4.5 内存管理
- [ ] 3D医学图像: 实现Patch采样
- [ ] 使用混合精度训练（AMP）
- [ ] DataLoader优化: pin_memory、persistent_workers

### 4.6 多模态处理（如适用）
- [ ] 实现多模态数据加载（T1/T2/FLAIR等）
- [ ] 实现模态融合策略（拼接/注意力/门控）
- [ ] 处理模态缺失情况

## 五、代码结构要求

### 5.1 必须实现的类/函数
```python
# 1. 配置类
@dataclass
class TrainingConfig:
    # 实验信息
    experiment_name: str
    seed: int = 42
    
    # 训练参数
    epochs: int
    batch_size: int
    learning_rate: float
    
    # 路径
    data_dir: str
    output_dir: str

# 2. 日志管理器
class ExperimentLogger:
    def log_environment(self): ...
    def log_config(self): ...
    def log_model(self, model): ...
    def log_epoch_start(self, epoch): ...
    def log_epoch_end(self, epoch, metrics): ...
    def log_batch(self, batch_idx, loss, lr): ...

# 3. Checkpoint管理器
class CheckpointManager:
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best): ...
    def load_checkpoint(self, model, optimizer, load_type='latest'): ...
    def transition_to_next_stage(self, next_stage_name): ...

# 4. 训练监控器
class TrainingMonitor:
    def on_epoch_start(self, epoch): ...
    def on_step_end(self, step, loss, lr): ...
    def on_epoch_end(self, metrics): ...
    def check_anomalies(self, loss, model): ...
```

### 5.2 训练循环结构
```python
def main():
    # 1. 初始化
    config = load_config()
    set_seed(config.seed)
    logger = ExperimentLogger(config.output_dir, config)
    
    # 2. 记录实验信息
    logger.log_environment()
    logger.log_config()
    
    # 3. 创建模型和数据
    model = create_model()
    logger.log_model(model)
    
    train_loader, val_loader = create_dataloaders(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 4. 创建Checkpoint和监控器
    ckpt_manager = CheckpointManager(config.output_dir, stage_info)
    monitor = TrainingMonitor(config.epochs, len(train_loader))
    
    # 5. 尝试恢复训练
    checkpoint = ckpt_manager.load_checkpoint(model, optimizer, scheduler)
    start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0
    
    # 6. 训练循环
    for epoch in range(start_epoch, config.epochs):
        monitor.on_epoch_start(epoch)
        
        # 训练阶段
        for step, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer)
            monitor.on_step_end(step, loss, optimizer.param_groups[0]['lr'])
            
            # 异常检测
            if monitor.check_anomalies(loss, model):
                logger.log_warning(f"Anomaly detected at epoch {epoch}, step {step}")
        
        # 验证阶段
        val_metrics = validate(model, val_loader)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 保存checkpoint
        is_best = val_metrics['accuracy'] > best_accuracy
        ckpt_manager.save_checkpoint(
            model, optimizer, scheduler,
            epoch, global_step, val_metrics,
            is_best=is_best
        )
        
        monitor.on_epoch_end(val_metrics)
    
    logger.log_training_summary()
```

=== 项目具体信息 ===

请根据以下项目信息生成完整代码:

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

请生成:
1. 完整的配置文件
2. 完整的训练脚本（包含上述所有规范）
3. 数据加载器（包含预处理和数据增强）
4. 模型定义
5. 损失函数和评估指标
6. 启动脚本

确保代码:
- 可以直接运行
- 包含详细的注释
- 符合上述所有检查清单
```

---

## 快速检查表（复制使用）

在代码审查时使用以下检查表：

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

*提示词版本: 1.0*  
*适用领域: CV / 医学图像*  
*框架: PyTorch*
