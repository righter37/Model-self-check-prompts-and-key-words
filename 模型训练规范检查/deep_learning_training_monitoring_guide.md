# 深度学习模型训练实时监控规范清单

## 目录
1. [必须实时输出的核心指标](#1-必须实时输出的核心指标)
2. [可视化工具推荐与选择建议](#2-可视化工具推荐与选择建议)
3. [控制台输出格式规范](#3-控制台输出格式规范)
4. [Python代码实现示例](#4-python代码实现示例)
5. [异常检测与告警机制](#5-异常检测与告警机制)
6. [最佳实践建议](#6-最佳实践建议)

---

## 1. 必须实时输出的核心指标

### 1.1 基础训练指标（必需）

| 指标名称 | 说明 | 输出频率 | 重要性 |
|---------|------|---------|--------|
| **Epoch** | 当前训练轮次 | 每轮 | ⭐⭐⭐ |
| **Step/Iteration** | 当前迭代步数 | 每步/每N步 | ⭐⭐⭐ |
| **Loss** | 当前批次损失值 | 每步 | ⭐⭐⭐ |
| **Learning Rate** | 当前学习率 | 每步/每轮 | ⭐⭐⭐ |
| **Time per Step** | 每步耗时 | 每步 | ⭐⭐ |
| **ETA** | 预计完成时间 | 每轮 | ⭐⭐ |

### 1.2 性能评估指标（推荐）

| 指标名称 | 适用场景 | 计算频率 |
|---------|---------|---------|
| **Accuracy** | 分类任务 | 每轮/每验证轮 |
| **Precision** | 分类任务（不平衡数据） | 每验证轮 |
| **Recall** | 分类任务（不平衡数据） | 每验证轮 |
| **F1-Score** | 分类任务综合评估 | 每验证轮 |
| **mAP** | 目标检测 | 每验证轮 |
| **BLEU/ROUGE** | NLP生成任务 | 每验证轮 |
| **Perplexity** | 语言模型 | 每验证轮 |

### 1.3 系统资源指标（可选但推荐）

| 指标名称 | 说明 | 监控工具 |
|---------|------|---------|
| **GPU Memory** | GPU显存使用 | nvidia-smi |
| **GPU Utilization** | GPU利用率 | nvidia-smi |
| **CPU Usage** | CPU使用率 | psutil |
| **RAM Usage** | 内存使用 | psutil |
| **Disk I/O** | 磁盘读写 | iostat |

### 1.4 高级训练指标（进阶）

| 指标名称 | 用途 | 实现复杂度 |
|---------|------|-----------|
| **Gradient Norm** | 检测梯度爆炸/消失 | 中 |
| **Weight Norm** | 监控参数变化 | 中 |
| **Activation Statistics** | 分析激活分布 | 高 |
| **Learning Rate Schedule** | 学习率变化曲线 | 低 |

---

## 2. 可视化工具推荐与选择建议

### 2.1 工具对比表

| 工具 | 推荐指数 | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|---------|
| **TensorBoard** | ⭐⭐⭐⭐⭐ | 免费、与PyTorch/TF深度集成、功能全面 | 界面相对简单 | 本地开发、中小型项目 |
| **Weights & Biases (wandb)** | ⭐⭐⭐⭐⭐ | 云端存储、协作友好、实验管理强 | 免费版有限制 | 团队协作、生产环境 |
| **MLflow** | ⭐⭐⭐⭐ | 开源、模型版本管理、部署支持 | 设置较复杂 | MLOps、模型管理 |
| **Neptune** | ⭐⭐⭐⭐ | 专业实验跟踪、可视化强 | 付费为主 | 大型团队、企业级 |
| **Comet** | ⭐⭐⭐ | 实时协作、调试工具 | 免费版限制多 | 快速原型开发 |
| **Visdom** | ⭐⭐⭐ | 轻量、灵活 | 功能较少 | 简单可视化需求 |

### 2.2 选择建议

```
决策流程：
┌─────────────────────────────────────────────────────────┐
│  是否需要团队协作？                                      │
│     ├── 是 → 选择 wandb / MLflow / Neptune              │
│     └── 否 → 继续判断                                    │
│                                                          │
│  是否需要模型版本管理？                                  │
│     ├── 是 → 选择 MLflow                                │
│     └── 否 → 继续判断                                    │
│                                                          │
│  预算是否充足？                                          │
│     ├── 是 → 选择 Neptune / Comet                       │
│     └── 否 → 选择 TensorBoard（免费）或 wandb免费版     │
└─────────────────────────────────────────────────────────┘
```

### 2.3 推荐配置

**个人开发者/学生**：TensorBoard（免费、功能足够）
**小团队**：wandb免费版（10GB存储，协作方便）
**企业级**：MLflow + 自托管 或 wandb企业版

---

## 3. 控制台输出格式规范

### 3.1 推荐输出格式

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

### 3.2 进度条格式（tqdm风格）

```
Epoch 3/100:  25%|███████████                            | 1250/5000 [02:37<07:51, 7.96it/s, loss=0.523, acc=0.873]
```

### 3.3 表格格式（多指标对比）

```
+--------+----------+----------+----------+----------+----------+--------+
| Epoch  | Train Loss | Val Loss | Train Acc | Val Acc  | LR       | Time   |
+--------+----------+----------+----------+----------+----------+--------+
| 1/100  | 1.2345   | 1.1234   | 65.32%   | 67.89%   | 1.00e-03 | 125s   |
| 2/100  | 0.8765   | 0.8234   | 78.45%   | 80.12%   | 1.00e-03 | 124s   |
| 3/100  | 0.5234   | 0.4891   | 87.25%   | 89.12%   | 1.00e-04 | 125s   |
+--------+----------+----------+----------+----------+----------+--------+
```

### 3.4 颜色编码建议

| 颜色 | 用途 | ANSI代码 |
|------|------|---------|
| 绿色 | 指标改善 | `\033[92m` |
| 红色 | 指标恶化/异常 | `\033[91m` |
| 黄色 | 警告/注意 | `\033[93m` |
| 蓝色 | 信息性内容 | `\033[94m` |
| 重置 | 结束颜色 | `\033[0m` |

---

## 4. Python代码实现示例

### 4.1 基础训练监控类

```python
import time
import torch
from datetime import timedelta
from collections import defaultdict
import sys


class TrainingMonitor:
    """训练监控器 - 基础版"""
    
    def __init__(self, total_epochs, steps_per_epoch, log_interval=10):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.log_interval = log_interval
        self.start_time = time.time()
        self.epoch_start_time = None
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        
    def on_epoch_start(self, epoch):
        """Epoch开始"""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_metrics = defaultdict(list)
        
    def on_step_end(self, step, loss, lr, **kwargs):
        """每步结束"""
        self.current_step = step
        self.epoch_metrics['loss'].append(loss)
        
        # 定期输出
        if step % self.log_interval == 0 or step == self.steps_per_epoch:
            self._print_step_info(step, loss, lr, **kwargs)
            
    def on_epoch_end(self, val_metrics=None):
        """Epoch结束"""
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.epoch_metrics['loss']) / len(self.epoch_metrics['loss'])
        
        # 保存历史
        self.metrics_history['train_loss'].append(avg_loss)
        self.metrics_history['epoch_time'].append(epoch_time)
        
        # 更新最佳指标
        if val_metrics:
            for key, value in val_metrics.items():
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
                    
        self._print_epoch_summary(avg_loss, val_metrics, epoch_time)
        
    def _print_step_info(self, step, loss, lr, **kwargs):
        """打印步骤信息"""
        progress = step / self.steps_per_epoch * 100
        eta = self._calculate_eta()
        
        print(f"\rEpoch [{self.current_epoch}/{self.total_epochs}] "
              f"Step [{step}/{self.steps_per_epoch}] ({progress:.1f}%) | "
              f"Loss: {loss:.4f} | LR: {lr:.2e} | ETA: {eta}", end='')
        sys.stdout.flush()
        
    def _print_epoch_summary(self, avg_loss, val_metrics, epoch_time):
        """打印Epoch摘要"""
        print(f"\n{'='*80}")
        print(f"Epoch [{self.current_epoch}/{self.total_epochs}] Summary:")
        print(f"  Train Loss: {avg_loss:.4f}")
        
        if val_metrics:
            for key, value in val_metrics.items():
                best = self.best_metrics.get(key, value)
                marker = "★" if value == best else " "
                print(f"  Val {key}: {value:.4f} {marker} (Best: {best:.4f})")
                
        print(f"  Time: {epoch_time:.1f}s | Total: {self._get_elapsed_time()}")
        print(f"{'='*80}\n")
        
    def _calculate_eta(self):
        """计算预计完成时间"""
        elapsed = time.time() - self.start_time
        progress = (self.current_epoch - 1) / self.total_epochs + \
                   self.current_step / (self.steps_per_epoch * self.total_epochs)
        if progress > 0:
            eta_seconds = elapsed / progress - elapsed
            return str(timedelta(seconds=int(eta_seconds)))
        return "N/A"
        
    def _get_elapsed_time(self):
        """获取已用时间"""
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))
```

### 4.2 TensorBoard集成

```python
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        step = step or self.global_step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        """记录多个标量"""
        step = step or self.global_step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_histogram(self, tag, values, step=None):
        """记录直方图（权重分布等）"""
        step = step or self.global_step
        self.writer.add_histogram(tag, values, step)
        
    def log_model_graph(self, model, input_sample):
        """记录模型结构"""
        self.writer.add_graph(model, input_sample)
        
    def log_image(self, tag, image, step=None):
        """记录图像"""
        step = step or self.global_step
        self.writer.add_image(tag, image, step)
        
    def log_hparams(self, hparams, metrics):
        """记录超参数"""
        self.writer.add_hparams(hparams, metrics)
        
    def close(self):
        self.writer.close()


# 使用示例
def train_with_tensorboard(model, train_loader, val_loader, epochs):
    logger = TensorBoardLogger('runs/my_experiment')
    monitor = TrainingMonitor(epochs, len(train_loader))
    
    for epoch in range(1, epochs + 1):
        monitor.on_epoch_start(epoch)
        
        for step, (data, target) in enumerate(train_loader, 1):
            # 训练步骤...
            loss = train_step(model, data, target)
            
            # 记录到TensorBoard
            logger.log_scalar('Loss/train_step', loss, step + (epoch-1)*len(train_loader))
            
            monitor.on_step_end(step, loss, optimizer.param_groups[0]['lr'])
            
        # 验证
        val_loss, val_acc = validate(model, val_loader)
        
        # 记录到TensorBoard
        logger.log_scalar('Loss/train_epoch', monitor.epoch_metrics['loss'][-1], epoch)
        logger.log_scalar('Loss/val', val_loss, epoch)
        logger.log_scalar('Accuracy/val', val_acc, epoch)
        logger.log_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录权重分布
        for name, param in model.named_parameters():
            logger.log_histogram(f'weights/{name}', param, epoch)
            if param.grad is not None:
                logger.log_histogram(f'gradients/{name}', param.grad, epoch)
        
        monitor.on_epoch_end({'loss': val_loss, 'accuracy': val_acc})
        
    logger.close()
```

### 4.3 Weights & Biases (wandb)集成

```python
import wandb


class WandbLogger:
    """Weights & Biases日志记录器"""
    
    def __init__(self, project_name, config=None, name=None):
        """
        Args:
            project_name: wandb项目名称
            config: 超参数字典
            name: 本次运行的名称
        """
        self.run = wandb.init(
            project=project_name,
            config=config,
            name=name,
            reinit=True
        )
        
    def log(self, data, step=None):
        """记录数据"""
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)
            
    def log_metrics(self, metrics, prefix='', step=None):
        """记录指标（带前缀）"""
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.log(prefixed_metrics, step)
        
    def watch_model(self, model, log='all', log_freq=100):
        """监控模型梯度和参数"""
        wandb.watch(model, log=log, log_freq=log_freq)
        
    def save_model(self, model_path, aliases=None):
        """保存模型到wandb"""
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(model_path)
        self.run.log_artifact(artifact, aliases=aliases)
        
    def finish(self):
        wandb.finish()


# 使用示例
def train_with_wandb(model, train_loader, val_loader, epochs):
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': epochs,
        'optimizer': 'Adam',
        'model': 'ResNet50'
    }
    
    logger = WandbLogger('my_project', config=config, name='experiment_1')
    logger.watch_model(model, log='all', log_freq=100)
    
    for epoch in range(1, epochs + 1):
        # 训练循环...
        train_metrics = train_epoch(model, train_loader)
        val_metrics = validate(model, val_loader)
        
        # 记录指标
        logger.log_metrics(train_metrics, prefix='train', step=epoch)
        logger.log_metrics(val_metrics, prefix='val', step=epoch)
        
        # 记录学习率
        logger.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)
        
    logger.finish()
```

### 4.4 带异常检测的增强版监控器

```python
import numpy as np
from typing import Callable, Optional


class AdvancedTrainingMonitor(TrainingMonitor):
    """增强版训练监控器 - 带异常检测"""
    
    def __init__(self, total_epochs, steps_per_epoch, log_interval=10,
                 anomaly_callbacks=None):
        super().__init__(total_epochs, steps_per_epoch, log_interval)
        
        self.anomaly_callbacks = anomaly_callbacks or {}
        self.loss_history = []
        self.gradient_norms = []
        
        # 异常阈值配置
        self.thresholds = {
            'loss_explosion': 10.0,      # loss爆炸阈值
            'loss_nan': True,             # 检测NaN
            'loss_stagnation': 50,        # loss停滞步数
            'grad_explosion': 100.0,      # 梯度爆炸阈值
            'grad_vanishing': 1e-7,       # 梯度消失阈值
        }
        
    def on_step_end(self, step, loss, lr, model=None, **kwargs):
        """增强版步骤结束处理"""
        # 异常检测
        self._check_anomalies(loss, model)
        
        # 记录历史
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
            
        super().on_step_end(step, loss, lr, **kwargs)
        
    def _check_anomalies(self, loss, model):
        """检查训练异常"""
        # 1. 检测NaN
        if self.thresholds['loss_nan'] and (np.isnan(loss) or np.isinf(loss)):
            self._trigger_anomaly('loss_nan', f'Loss is NaN or Inf: {loss}')
            return
            
        # 2. 检测loss爆炸
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if loss > recent_avg * self.thresholds['loss_explosion']:
                self._trigger_anomaly('loss_explosion', 
                    f'Loss explosion detected: {loss:.4f} (avg: {recent_avg:.4f})')
                
        # 3. 检测loss停滞
        if len(self.loss_history) >= self.thresholds['loss_stagnation']:
            recent_std = np.std(self.loss_history[-self.thresholds['loss_stagnation']:])
            if recent_std < 1e-6:
                self._trigger_anomaly('loss_stagnation',
                    f'Loss has stagnated for {self.thresholds["loss_stagnation"]} steps')
                    
        # 4. 检测梯度异常
        if model is not None:
            self._check_gradient_anomalies(model)
            
    def _check_gradient_anomalies(self, model):
        """检测梯度异常"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # 检测单个参数梯度爆炸
                if param_norm > self.thresholds['grad_explosion']:
                    self._trigger_anomaly('grad_explosion',
                        f'Gradient explosion: norm={param_norm:.4f}')
                    
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)
        
        # 检测梯度消失
        if total_norm < self.thresholds['grad_vanishing']:
            self._trigger_anomaly('grad_vanishing',
                f'Gradient vanishing: norm={total_norm:.2e}')
                
    def _trigger_anomaly(self, anomaly_type, message):
        """触发异常处理"""
        warning_msg = f"\n⚠️  [ANOMALY] {anomaly_type}: {message}"
        print(f"\033[91m{warning_msg}\033[0m")
        
        # 执行回调
        if anomaly_type in self.anomaly_callbacks:
            self.anomaly_callbacks[anomaly_type](message)
```

### 4.5 系统资源监控

```python
import psutil
import pynvml


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
    def get_system_info(self):
        """获取系统信息"""
        info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        }
        
        if self.has_gpu:
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            info.update({
                'gpu_memory_used_gb': gpu_info.used / (1024**3),
                'gpu_memory_total_gb': gpu_info.total / (1024**3),
                'gpu_memory_percent': gpu_info.used / gpu_info.total * 100,
                'gpu_utilization': gpu_util.gpu,
            })
            
        return info
        
    def print_system_info(self):
        """打印系统信息"""
        info = self.get_system_info()
        print(f"CPU: {info['cpu_percent']:.1f}% | "
              f"RAM: {info['memory_used_gb']:.1f}GB ({info['memory_percent']:.1f}%)")
        if self.has_gpu:
            print(f"GPU: {info['gpu_utilization']}% | "
                  f"VRAM: {info['gpu_memory_used_gb']:.1f}/{info['gpu_memory_total_gb']:.1f}GB "
                  f"({info['gpu_memory_percent']:.1f}%)")
```

---

## 5. 异常检测与告警机制

### 5.1 异常类型与检测方法

| 异常类型 | 检测条件 | 严重程度 | 建议操作 |
|---------|---------|---------|---------|
| **Loss NaN/Inf** | `loss != loss` 或 `loss == inf` | 🔴 严重 | 立即停止，检查数据/学习率 |
| **Loss爆炸** | loss > 历史均值 × 10 | 🔴 严重 | 降低学习率，检查梯度裁剪 |
| **Loss停滞** | 连续50步loss变化 < 1e-6 | 🟡 警告 | 调整学习率，检查数据 |
| **梯度爆炸** | grad_norm > 100 | 🔴 严重 | 启用梯度裁剪 |
| **梯度消失** | grad_norm < 1e-7 | 🟡 警告 | 检查激活函数、初始化 |
| **过拟合** | val_loss上升，train_loss下降 | 🟡 警告 | 增加正则化、早停 |
| **GPU OOM** | CUDA out of memory | 🔴 严重 | 减小batch size |
| **学习率过小** | lr < 1e-8 且 loss不下降 | 🟡 警告 | 增大学习率 |

### 5.2 告警通知实现

```python
import smtplib
from email.mime.text import MIMEText
import requests


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.alert_history = []
        
    def send_email(self, subject, body, to_email):
        """发送邮件告警"""
        if not self.config.get('email'):
            return
            
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.config['email']['from']
        msg['To'] = to_email
        
        with smtplib.SMTP(self.config['email']['smtp_server']) as server:
            server.login(self.config['email']['username'], 
                        self.config['email']['password'])
            server.send_message(msg)
            
    def send_slack(self, message, webhook_url):
        """发送Slack通知"""
        payload = {'text': message}
        requests.post(webhook_url, json=payload)
        
    def send_wechat(self, message, webhook_key):
        """发送企业微信通知"""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
        payload = {
            "msgtype": "text",
            "text": {"content": message}
        }
        requests.post(url, json=payload)
        
    def alert(self, level, message, notify=False):
        """
        触发告警
        level: 'info', 'warning', 'error', 'critical'
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        alert_msg = f"[{timestamp}] [{level.upper()}] {message}"
        
        # 颜色输出
        colors = {
            'info': '\033[94m',
            'warning': '\033[93m',
            'error': '\033[91m',
            'critical': '\033[91m\033[1m'
        }
        print(f"{colors.get(level, '')}{alert_msg}\033[0m")
        
        self.alert_history.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
        
        # 发送通知
        if notify and level in ['error', 'critical']:
            self._send_notifications(alert_msg)
```

### 5.3 早停机制

```python
class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: 容忍轮数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"\n⏹️  Early stopping triggered after {self.patience} epochs without improvement")
            
        return self.early_stop
```

---

## 6. 最佳实践建议

### 6.1 监控频率建议

| 指标类型 | 记录频率 | 输出频率 | 说明 |
|---------|---------|---------|------|
| Loss | 每步 | 每10-100步 | 避免输出过于频繁 |
| Accuracy | 每轮 | 每轮 | 评估指标 |
| 学习率 | 每步 | 每轮 | 跟踪调度 |
| 梯度范数 | 每步 | 每轮 | 调试时使用 |
| 系统资源 | 每30秒 | 每轮 | 监控资源使用 |

### 6.2 推荐的监控配置

```python
# 推荐的监控配置（适用于大多数场景）
MONITOR_CONFIG = {
    # 日志设置
    'log_interval': 50,           # 每50步输出一次
    'log_to_tensorboard': True,
    'log_to_wandb': False,        # 根据需要启用
    
    # 异常检测
    'enable_anomaly_detection': True,
    'loss_explosion_factor': 10.0,
    'grad_clip_value': 1.0,
    
    # 早停
    'early_stopping': {
        'enabled': True,
        'patience': 10,
        'min_delta': 0.001
    },
    
    # 检查点
    'checkpoint': {
        'save_interval': 5,       # 每5轮保存
        'save_best_only': True,
        'monitor': 'val_loss'
    },
    
    # 系统监控
    'monitor_system': True,
    'system_log_interval': 60     # 每60秒
}
```

### 6.3 完整训练循环示例

```python
def complete_training_loop(model, train_loader, val_loader, config):
    """完整的训练循环示例"""
    
    # 初始化
    monitor = AdvancedTrainingMonitor(
        total_epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        log_interval=config['log_interval']
    )
    
    tb_logger = TensorBoardLogger(config['log_dir']) if config['log_to_tensorboard'] else None
    early_stopping = EarlyStopping(**config['early_stopping']) if config['early_stopping']['enabled'] else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['epochs'] + 1):
        monitor.on_epoch_start(epoch)
        model.train()
        
        # 训练阶段
        for step, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_value'])
            
            optimizer.step()
            
            # 监控
            monitor.on_step_end(step, loss.item(), optimizer.param_groups[0]['lr'], model=model)
            
            if tb_logger:
                tb_logger.log_scalar('train/loss_step', loss.item(), 
                                   step + (epoch-1)*len(train_loader))
        
        # 验证阶段
        model.eval()
        val_loss, val_acc = validate(model, val_loader)
        
        # 记录到TensorBoard
        if tb_logger:
            tb_logger.log_scalar('train/loss_epoch', 
                               sum(monitor.epoch_metrics['loss'])/len(monitor.epoch_metrics['loss']), 
                               epoch)
            tb_logger.log_scalar('val/loss', val_loss, epoch)
            tb_logger.log_scalar('val/accuracy', val_acc, epoch)
            tb_logger.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ Saved best model with val_loss: {val_loss:.4f}")
        
        # 早停检查
        if early_stopping and early_stopping(val_loss):
            break
            
        monitor.on_epoch_end({'loss': val_loss, 'accuracy': val_acc})
    
    if tb_logger:
        tb_logger.close()
```

### 6.4 快速启动模板

```python
# 快速启动模板 - 复制即可使用
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class SimpleTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter()
        self.global_step = 0
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 记录到TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
        accuracy = correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy
    
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/accuracy', val_acc, epoch)
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
                  
        self.writer.close()

# 使用示例
# trainer = SimpleTrainer(model, optimizer)
# trainer.fit(train_loader, val_loader, epochs=100)
```

---

## 附录：常用命令速查

### TensorBoard启动
```bash
# 基本启动
tensorboard --logdir=runs

# 指定端口
tensorboard --logdir=runs --port=6006

# 远程访问
tensorboard --logdir=runs --bind_all
```

### wandb常用命令
```bash
# 登录
wandb login

# 查看运行
wandb runs

# 同步离线运行
wandb sync wandb/offline-run-*
```

### 监控GPU
```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细监控
nvidia-smi dmon -s u
```

---

*文档版本: 1.0*  
*最后更新: 2024年*
