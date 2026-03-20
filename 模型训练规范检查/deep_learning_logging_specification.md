# 深度学习模型训练日志记录规范清单

## 目录
1. [概述](#概述)
2. [必须记录的信息类别](#必须记录的信息类别)
3. [日志存储结构建议](#日志存储结构建议)
4. [完整代码示例](#完整代码示例)
5. [最佳实践建议](#最佳实践建议)

---

## 概述

完整的训练日志记录是深度学习实验可复现性的基础。良好的日志系统应能回答：
- **发生了什么？** - 训练过程中的所有事件
- **何时发生的？** - 精确的时间戳
- **为什么发生？** - 参数配置和环境信息
- **如何重现？** - 完整的实验配置

---

## 必须记录的信息类别

### 1. 基础环境信息

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| Python版本 | ⭐⭐⭐⭐⭐ | 不同版本可能导致API行为差异 |
| PyTorch/TensorFlow版本 | ⭐⭐⭐⭐⭐ | 框架版本影响模型行为和性能 |
| CUDA版本 | ⭐⭐⭐⭐⭐ | 影响GPU计算结果的一致性 |
| GPU型号与数量 | ⭐⭐⭐⭐ | 硬件差异影响训练速度和内存 |
| 操作系统 | ⭐⭐⭐⭐ | 系统调用和环境变量差异 |
| 依赖包版本 | ⭐⭐⭐⭐ | 第三方库版本兼容性 |

```python
import sys
import platform
import torch
import logging
from datetime import datetime

def log_environment_info(logger):
    """记录环境信息"""
    logger.info("=" * 50)
    logger.info("环境信息")
    logger.info("=" * 50)
    
    # Python环境
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"Python可执行路径: {sys.executable}")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"机器架构: {platform.machine()}")
    
    # PyTorch环境
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i}显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # 记录当前时间
    logger.info(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
```

---

### 2. 随机种子与可复现性

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 随机种子值 | ⭐⭐⭐⭐⭐ | 确保实验可完全复现 |
| 确定性设置 | ⭐⭐⭐⭐⭐ | 控制非确定性操作 |

```python
import random
import numpy as np
import torch

def set_random_seed(seed: int, logger: logging.Logger = None):
    """设置随机种子并记录"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if logger:
        logger.info(f"随机种子已设置: {seed}")
        logger.info(f"CuDNN确定性模式: {torch.backends.cudnn.deterministic}")
        logger.info(f"CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    return seed
```

---

### 3. 超参数配置

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 学习率 | ⭐⭐⭐⭐⭐ | 影响收敛速度和效果 |
| 批量大小 | ⭐⭐⭐⭐⭐ | 影响内存使用和梯度估计 |
| 优化器类型 | ⭐⭐⭐⭐⭐ | 决定参数更新策略 |
| 学习率调度 | ⭐⭐⭐⭐⭐ | 影响训练动态 |
| 正则化参数 | ⭐⭐⭐⭐ | 防止过拟合 |
| 训练轮数 | ⭐⭐⭐⭐⭐ | 训练时长控制 |

```python
from dataclasses import dataclass, asdict
import json

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 优化器参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "Adam"
    
    # 训练参数
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    
    # 学习率调度
    lr_scheduler: str = "StepLR"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # 正则化
    dropout: float = 0.5
    label_smoothing: float = 0.0
    
    # 其他
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 1

def log_config(config: TrainingConfig, logger: logging.Logger, save_path: str = None):
    """记录配置信息"""
    config_dict = asdict(config)
    
    logger.info("=" * 50)
    logger.info("训练配置")
    logger.info("=" * 50)
    
    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")
    
    # 保存为JSON文件
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        logger.info(f"配置已保存至: {save_path}")
    
    logger.info("=" * 50)
```

---

### 4. 模型架构信息

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 模型结构 | ⭐⭐⭐⭐⭐ | 理解模型设计 |
| 参数量 | ⭐⭐⭐⭐⭐ | 评估模型复杂度 |
| 模型大小 | ⭐⭐⭐⭐ | 部署和存储考量 |
| 各层输出维度 | ⭐⭐⭐⭐ | 调试维度错误 |

```python
def log_model_info(model: torch.nn.Module, logger: logging.Logger, input_shape: tuple = None):
    """记录模型信息"""
    logger.info("=" * 50)
    logger.info("模型架构信息")
    logger.info("=" * 50)
    
    # 模型结构
    logger.info("模型结构:")
    logger.info(str(model))
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info(f"不可训练参数量: {total_params - trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 各层参数详情
    logger.info("各层参数详情:")
    for name, param in model.named_parameters():
        logger.info(f"  {name}: {param.shape} ({param.numel():,} 参数)")
    
    # 测试前向传播
    if input_shape is not None:
        logger.info(f"输入形状: {input_shape}")
        try:
            dummy_input = torch.randn(*input_shape)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            with torch.no_grad():
                output = model(dummy_input)
            logger.info(f"输出形状: {output.shape}")
        except Exception as e:
            logger.error(f"前向传播测试失败: {e}")
    
    logger.info("=" * 50)
```

---

### 5. 数据预处理与数据加载

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 数据增强策略 | ⭐⭐⭐⭐⭐ | 影响模型泛化能力 |
| 归一化参数 | ⭐⭐⭐⭐⭐ | 影响输入分布 |
| 数据集划分 | ⭐⭐⭐⭐⭐ | 确保实验可复现 |
| 数据加载配置 | ⭐⭐⭐⭐ | 影响训练效率 |

```python
def log_data_info(train_dataset, val_dataset, test_dataset, 
                  transform, logger: logging.Logger):
    """记录数据信息"""
    logger.info("=" * 50)
    logger.info("数据集信息")
    logger.info("=" * 50)
    
    # 数据集大小
    logger.info(f"训练集大小: {len(train_dataset) if train_dataset else 0}")
    logger.info(f"验证集大小: {len(val_dataset) if val_dataset else 0}")
    logger.info(f"测试集大小: {len(test_dataset) if test_dataset else 0}")
    
    # 数据增强
    logger.info("数据预处理流程:")
    logger.info(str(transform))
    
    # 类别分布（如果是分类任务）
    if hasattr(train_dataset, 'classes'):
        logger.info(f"类别数: {len(train_dataset.classes)}")
        logger.info(f"类别名称: {train_dataset.classes}")
    
    # 样本形状
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        if isinstance(sample, tuple):
            logger.info(f"样本输入形状: {sample[0].shape if hasattr(sample[0], 'shape') else type(sample[0])}")
            logger.info(f"样本标签类型: {type(sample[1])}")
    
    logger.info("=" * 50)
```

---

### 6. 训练过程指标

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 损失值 | ⭐⭐⭐⭐⭐ | 监控训练进度 |
| 准确率/指标 | ⭐⭐⭐⭐⭐ | 评估模型性能 |
| 学习率 | ⭐⭐⭐⭐⭐ | 验证调度策略 |
| 训练时间 | ⭐⭐⭐⭐ | 资源规划 |
| GPU显存使用 | ⭐⭐⭐⭐ | 防止OOM |
| 梯度范数 | ⭐⭐⭐⭐ | 检测梯度问题 |

```python
import time
from collections import defaultdict

class TrainingLogger:
    """训练过程日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_times = []
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Epoch [{epoch}/{total_epochs}] 开始")
        self.logger.info(f"{'='*50}")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        self.logger.info(f"\nEpoch [{epoch}] 完成 - 耗时: {epoch_time:.2f}s")
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # 记录GPU显存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                self.logger.info(f"  GPU {i} 显存使用: {allocated:.2f}GB / {reserved:.2f}GB")
    
    def log_batch(self, batch_idx: int, total_batches: int, 
                  loss: float, lr: float, log_interval: int = 10):
        """记录批次信息"""
        if batch_idx % log_interval == 0:
            self.logger.info(
                f"  Batch [{batch_idx}/{total_batches}] "
                f"Loss: {loss:.6f} LR: {lr:.8f}"
            )
    
    def log_gradient_norm(self, model: torch.nn.Module):
        """记录梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.info(f"  梯度范数: {total_norm:.6f}")
        return total_norm
    
    def log_training_summary(self):
        """记录训练总结"""
        total_time = time.time() - self.start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info("训练总结")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"总训练时间: {total_time/3600:.2f} 小时")
        self.logger.info(f"平均每个epoch时间: {avg_epoch_time/60:.2f} 分钟")
        self.logger.info(f"总epoch数: {len(self.epoch_times)}")
        
        # 最佳指标
        for key, values in self.metrics_history.items():
            if 'loss' in key.lower():
                best_value = min(values)
                best_epoch = values.index(best_value) + 1
                self.logger.info(f"最佳 {key}: {best_value:.6f} (Epoch {best_epoch})")
            elif 'acc' in key.lower() or 'score' in key.lower():
                best_value = max(values)
                best_epoch = values.index(best_value) + 1
                self.logger.info(f"最佳 {key}: {best_value:.6f} (Epoch {best_epoch})")
```

---

### 7. 验证与测试指标

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 验证损失 | ⭐⭐⭐⭐⭐ | 检测过拟合 |
| 验证准确率 | ⭐⭐⭐⭐⭐ | 模型选择依据 |
| 混淆矩阵 | ⭐⭐⭐⭐ | 分析错误模式 |
| 各类别指标 | ⭐⭐⭐⭐ | 发现类别不平衡问题 |
| 测试集结果 | ⭐⭐⭐⭐⭐ | 最终性能评估 |

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def log_evaluation_results(y_true, y_pred, class_names, logger: logging.Logger, 
                           dataset_name: str = "Validation"):
    """记录评估结果"""
    logger.info(f"\n{'='*50}")
    logger.info(f"{dataset_name} 评估结果")
    logger.info(f"{'='*50}")
    
    # 准确率
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    logger.info(f"准确率: {accuracy:.6f}")
    
    # 详细分类报告
    logger.info("\n分类报告:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    logger.info("\n" + report)
    
    # 混淆矩阵
    logger.info("混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    logger.info("\n" + str(cm))
    
    # 保存混淆矩阵为CSV格式
    cm_str = ",".join(class_names) + "\n"
    for i, row in enumerate(cm):
        cm_str += class_names[i] + "," + ",".join(map(str, row)) + "\n"
    logger.info(f"混淆矩阵(CSV格式):\n{cm_str}")
    
    return accuracy
```

---

### 8. 检查点与模型保存

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 保存路径 | ⭐⭐⭐⭐⭐ | 模型恢复 |
| 保存时间 | ⭐⭐⭐⭐ | 版本管理 |
| 对应指标 | ⭐⭐⭐⭐⭐ | 模型选择 |
| 优化器状态 | ⭐⭐⭐⭐ | 恢复训练 |

```python
def save_checkpoint(model, optimizer, epoch, metrics, config, save_path, logger):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    torch.save(checkpoint, save_path)
    
    logger.info(f"\n检查点已保存:")
    logger.info(f"  路径: {save_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  指标: {metrics}")
    logger.info(f"  时间: {checkpoint['timestamp']}")
    
    return save_path
```

---

### 9. 异常与错误信息

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 异常类型 | ⭐⭐⭐⭐⭐ | 问题诊断 |
| 错误堆栈 | ⭐⭐⭐⭐⭐ | 定位问题 |
| 上下文信息 | ⭐⭐⭐⭐ | 复现问题 |

```python
import traceback

def log_exception(logger: logging.Logger, e: Exception, context: dict = None):
    """记录异常信息"""
    logger.error(f"\n{'='*50}")
    logger.error("发生异常")
    logger.error(f"{'='*50}")
    logger.error(f"异常类型: {type(e).__name__}")
    logger.error(f"异常信息: {str(e)}")
    
    if context:
        logger.error("上下文信息:")
        for key, value in context.items():
            logger.error(f"  {key}: {value}")
    
    # 记录完整堆栈
    logger.error("堆栈跟踪:")
    logger.error(traceback.format_exc())
    logger.error(f"{'='*50}")
```

---

### 10. 实验元数据

| 记录项目 | 重要性 | 记录原因 |
|---------|--------|---------|
| 实验ID | ⭐⭐⭐⭐⭐ | 唯一标识实验 |
| 实验名称 | ⭐⭐⭐⭐ | 实验描述 |
| 实验目的 | ⭐⭐⭐⭐ | 实验背景 |
| 作者信息 | ⭐⭐⭐ | 责任追溯 |
| Git版本 | ⭐⭐⭐⭐⭐ | 代码版本控制 |

```python
import subprocess

def log_experiment_metadata(logger: logging.Logger, experiment_name: str, 
                            experiment_description: str = "", author: str = ""):
    """记录实验元数据"""
    logger.info(f"\n{'='*50}")
    logger.info("实验元数据")
    logger.info(f"{'='*50}")
    
    # 实验基本信息
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"实验描述: {experiment_description}")
    logger.info(f"作者: {author}")
    logger.info(f"实验ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Git信息
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        git_status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
        
        logger.info(f"Git分支: {git_branch}")
        logger.info(f"Git提交: {git_commit}")
        logger.info(f"Git状态: {'有未提交更改' if git_status else '干净'}")
        if git_status:
            logger.info(f"未提交更改:\n{git_status}")
    except:
        logger.info("Git信息: 无法获取（可能不在Git仓库中）")
    
    logger.info(f"{'='*50}")
```

---

## 日志存储结构建议

### 推荐的目录结构

```
experiments/
├── experiment_20241201_143052/          # 实验根目录（时间戳命名）
│   ├── logs/                            # 日志文件
│   │   ├── train.log                    # 训练日志（主日志）
│   │   ├── error.log                    # 错误日志
│   │   └── metrics.json                 # 结构化指标数据
│   ├── checkpoints/                     # 模型检查点
│   │   ├── best_model.pth               # 最佳模型
│   │   ├── checkpoint_epoch_10.pth      # 定期保存的检查点
│   │   └── checkpoint_epoch_20.pth
│   ├── config/                          # 配置文件
│   │   ├── config.json                  # 训练配置
│   │   └── model_architecture.txt       # 模型架构
│   ├── results/                         # 实验结果
│   │   ├── training_curves.png          # 训练曲线图
│   │   ├── confusion_matrix.png         # 混淆矩阵图
│   │   └── predictions.csv              # 预测结果
│   └── metadata/                        # 元数据
│       ├── git_info.txt                 # Git信息
│       └── environment.yaml             # 环境配置
```

### 日志文件配置

```python
import logging
import os
from datetime import datetime

def setup_logging(log_dir: str, experiment_name: str = None):
    """设置日志系统"""
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成实验名称
    if experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 主日志记录器
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    
    # 清除已有处理器
    logger.handlers = []
    
    # 文件处理器 - 记录所有日志
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'train.log'), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 文件处理器 - 仅记录错误
    error_handler = logging.FileHandler(
        os.path.join(log_dir, 'error.log'),
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger, experiment_name
```

---

## 完整代码示例

### 完整的训练日志系统集成

```python
"""
深度学习训练日志系统 - 完整示例
"""

import os
import sys
import json
import time
import random
import logging
import platform
import subprocess
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============== 配置类 ==============

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验信息
    experiment_name: str = "my_experiment"
    experiment_description: str = ""
    author: str = ""
    
    # 训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 系统参数
    num_workers: int = 4
    device: str = "auto"  # auto, cuda, cpu
    seed: int = 42
    
    # 日志参数
    log_interval: int = 10
    save_interval: int = 10
    
    def to_dict(self):
        return asdict(self)


# ============== 日志管理器 ==============

class ExperimentLogger:
    """实验日志管理器"""
    
    def __init__(self, log_dir: str, config: ExperimentConfig):
        self.config = config
        self.log_dir = log_dir
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = time.time()
        
        # 创建目录结构
        self._create_directory_structure()
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 指标历史
        self.metrics_history = defaultdict(list)
        
    def _create_directory_structure(self):
        """创建实验目录结构"""
        self.dirs = {
            'logs': os.path.join(self.log_dir, 'logs'),
            'checkpoints': os.path.join(self.log_dir, 'checkpoints'),
            'config': os.path.join(self.log_dir, 'config'),
            'results': os.path.join(self.log_dir, 'results'),
            'metadata': os.path.join(self.log_dir, 'metadata')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(self.config.experiment_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 主日志文件
        file_handler = logging.FileHandler(
            os.path.join(self.dirs['logs'], 'train.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 错误日志文件
        error_handler = logging.FileHandler(
            os.path.join(self.dirs['logs'], 'error.log'),
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_environment(self):
        """记录环境信息"""
        self.logger.info("=" * 60)
        self.logger.info("环境信息")
        self.logger.info("=" * 60)
        
        # Python环境
        self.logger.info(f"Python版本: {sys.version}")
        self.logger.info(f"操作系统: {platform.system()} {platform.release()}")
        self.logger.info(f"机器架构: {platform.machine()}")
        
        # PyTorch环境
        self.logger.info(f"PyTorch版本: {torch.__version__}")
        self.logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA版本: {torch.version.cuda}")
            self.logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
            self.logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                self.logger.info(f"  显存: {props.total_memory / 1e9:.2f} GB")
                self.logger.info(f"  计算能力: {props.major}.{props.minor}")
        
        # 保存环境配置
        self._save_environment_yaml()
        
        self.logger.info("=" * 60)
    
    def _save_environment_yaml(self):
        """保存环境配置为YAML"""
        try:
            import pkg_resources
            installed_packages = [
                f"{dist.key}=={dist.version}"
                for dist in pkg_resources.working_set
            ]
            
            env_content = f"""name: {self.config.experiment_name}
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python={platform.python_version()}
  - pytorch={torch.__version__}
pip:
"""
            for pkg in sorted(installed_packages):
                env_content += f"  - {pkg}\n"
            
            with open(os.path.join(self.dirs['metadata'], 'environment.yaml'), 'w') as f:
                f.write(env_content)
        except:
            pass
    
    def log_config(self):
        """记录配置信息"""
        self.logger.info("=" * 60)
        self.logger.info("实验配置")
        self.logger.info("=" * 60)
        
        for key, value in self.config.to_dict().items():
            self.logger.info(f"{key}: {value}")
        
        # 保存配置JSON
        config_path = os.path.join(self.dirs['config'], 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"配置已保存: {config_path}")
        self.logger.info("=" * 60)
    
    def log_model(self, model: nn.Module, input_shape: tuple = None):
        """记录模型信息"""
        self.logger.info("=" * 60)
        self.logger.info("模型架构信息")
        self.logger.info("=" * 60)
        
        # 模型结构
        self.logger.info("模型结构:")
        self.logger.info(str(model))
        
        # 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数量: {trainable_params:,}")
        self.logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 保存模型架构
        arch_path = os.path.join(self.dirs['config'], 'model_architecture.txt')
        with open(arch_path, 'w', encoding='utf-8') as f:
            f.write(str(model))
            f.write(f"\n\n总参数量: {total_params:,}\n")
            f.write(f"可训练参数量: {trainable_params:,}\n")
        
        self.logger.info("=" * 60)
    
    def log_experiment_metadata(self):
        """记录实验元数据"""
        self.logger.info("=" * 60)
        self.logger.info("实验元数据")
        self.logger.info("=" * 60)
        
        self.logger.info(f"实验ID: {self.experiment_id}")
        self.logger.info(f"实验名称: {self.config.experiment_name}")
        self.logger.info(f"实验描述: {self.config.experiment_description}")
        self.logger.info(f"作者: {self.config.author}")
        self.logger.info(f"开始时间: {datetime.fromtimestamp(self.start_time)}")
        
        # Git信息
        try:
            git_info = {
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'branch': subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip(),
                'status': subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
            }
            
            self.logger.info(f"Git分支: {git_info['branch']}")
            self.logger.info(f"Git提交: {git_info['commit']}")
            
            # 保存Git信息
            with open(os.path.join(self.dirs['metadata'], 'git_info.txt'), 'w') as f:
                for key, value in git_info.items():
                    f.write(f"{key}: {value}\n")
        except:
            self.logger.info("Git信息: 无法获取")
        
        self.logger.info("=" * 60)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.epoch_start_time = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch [{epoch}/{total_epochs}] 开始")
        self.logger.info(f"{'='*60}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """记录epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        
        self.logger.info(f"\nEpoch [{epoch}] 完成 - 耗时: {epoch_time:.2f}s")
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            self.logger.info(f"  {key}: {value:.6f}")
        
        # 记录GPU显存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                self.logger.info(f"  GPU {i} 显存: {allocated:.2f}GB / {reserved:.2f}GB")
        
        # 保存指标
        self._save_metrics()
    
    def log_batch(self, batch_idx: int, total_batches: int, 
                  loss: float, lr: float):
        """记录批次信息"""
        if batch_idx % self.config.log_interval == 0:
            self.logger.info(
                f"  Batch [{batch_idx}/{total_batches}] "
                f"Loss: {loss:.6f} LR: {lr:.8f}"
            )
    
    def log_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """记录检查点保存"""
        self.logger.info(f"\n检查点已保存 (Epoch {epoch})")
        self.logger.info(f"  指标: {metrics}")
        self.logger.info(f"  最佳模型: {is_best}")
    
    def log_exception(self, e: Exception, context: dict = None):
        """记录异常"""
        self.logger.error(f"\n{'='*60}")
        self.logger.error("发生异常")
        self.logger.error(f"{'='*60}")
        self.logger.error(f"异常类型: {type(e).__name__}")
        self.logger.error(f"异常信息: {str(e)}")
        
        if context:
            self.logger.error("上下文信息:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")
        
        self.logger.error("堆栈跟踪:")
        self.logger.error(traceback.format_exc())
        self.logger.error(f"{'='*60}")
    
    def _save_metrics(self):
        """保存指标到JSON"""
        metrics_path = os.path.join(self.dirs['logs'], 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.metrics_history), f, indent=4)
    
    def log_training_summary(self):
        """记录训练总结"""
        total_time = time.time() - self.start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练总结")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"总训练时间: {total_time/3600:.2f} 小时")
        self.logger.info(f"总epoch数: {len(self.metrics_history.get('train_loss', []))}")
        
        # 最佳指标
        for key, values in self.metrics_history.items():
            if values:
                if 'loss' in key.lower():
                    best_value = min(values)
                    best_epoch = values.index(best_value) + 1
                    self.logger.info(f"最佳 {key}: {best_value:.6f} (Epoch {best_epoch})")
                elif 'acc' in key.lower() or 'score' in key.lower():
                    best_value = max(values)
                    best_epoch = values.index(best_value) + 1
                    self.logger.info(f"最佳 {key}: {best_value:.6f} (Epoch {best_epoch})")
        
        self.logger.info(f"{'='*60}")


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    # 配置
    config = ExperimentConfig(
        experiment_name="resnet50_cifar10",
        experiment_description="使用ResNet50在CIFAR-10上训练",
        author="Researcher",
        epochs=10,
        batch_size=64,
        learning_rate=0.001,
        seed=42
    )
    
    # 创建日志管理器
    log_dir = os.path.join("experiments", f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    exp_logger = ExperimentLogger(log_dir, config)
    
    # 记录实验信息
    exp_logger.log_experiment_metadata()
    exp_logger.log_environment()
    exp_logger.log_config()
    
    # 创建简单模型并记录
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )
    exp_logger.log_model(model, input_shape=(1, 784))
    
    # 模拟训练
    for epoch in range(1, config.epochs + 1):
        exp_logger.log_epoch_start(epoch, config.epochs)
        
        # 模拟训练批次
        for batch_idx in range(100):
            loss = 1.0 / (epoch + batch_idx * 0.01)
            lr = config.learning_rate * (0.9 ** epoch)
            exp_logger.log_batch(batch_idx, 100, loss, lr)
        
        # 模拟验证指标
        metrics = {
            'train_loss': 0.5 / epoch,
            'train_acc': 0.7 + 0.02 * epoch,
            'val_loss': 0.6 / epoch,
            'val_acc': 0.65 + 0.015 * epoch
        }
        
        exp_logger.log_epoch_end(epoch, metrics)
    
    # 训练总结
    exp_logger.log_training_summary()
    
    print(f"\n实验日志已保存至: {log_dir}")


if __name__ == "__main__":
    example_usage()
```

---

## 最佳实践建议

### 1. 日志级别使用规范

| 级别 | 使用场景 |
|------|---------|
| DEBUG | 详细的调试信息，如每批次损失 |
| INFO | 重要事件，如epoch开始/结束 |
| WARNING | 潜在问题，如学习率过小 |
| ERROR | 错误但不终止程序 |
| CRITICAL | 严重错误，需要终止程序 |

### 2. 日志记录时机

```python
# ✅ 推荐：在关键节点记录
- 实验开始时记录所有配置
- 每个epoch开始和结束时记录
- 模型保存时记录
- 异常发生时记录
- 训练结束时记录总结

# ❌ 避免：过度记录
- 每个参数更新都记录
- 在循环中频繁记录大量数据
```

### 3. 日志文件管理

```python
# 日志轮转（防止日志文件过大）
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'train.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

### 4. 结构化日志

```python
# 使用JSON格式存储结构化数据
import json

def log_structured(logger, event_type: str, data: dict):
    """记录结构化日志"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'data': data
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
```

### 5. 与实验管理工具集成

```python
# TensorBoard集成
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', acc, epoch)

# Weights & Biases集成
import wandb

wandb.init(project="my-project", config=config.to_dict())
wandb.log({"loss": loss, "accuracy": acc})
```

---

## 总结

完整的深度学习训练日志应包含：

1. **环境信息** - Python、PyTorch、CUDA版本等
2. **随机种子** - 确保实验可复现
3. **超参数配置** - 所有训练参数
4. **模型架构** - 结构、参数量、大小
5. **数据信息** - 预处理、增强、划分
6. **训练指标** - 损失、准确率、学习率等
7. **验证指标** - 验证集性能
8. **检查点信息** - 模型保存记录
9. **异常信息** - 错误和堆栈跟踪
10. **实验元数据** - 实验ID、Git版本等

良好的日志系统是实验可复现性的基础，也是深度学习工程化的重要组成部分。

---

*文档生成时间: 2024年*
