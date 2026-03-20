# 深度学习Checkpoint管理规范清单

> 版本: 1.0 | 适用框架: PyTorch (可扩展至其他框架)

---

## 一、Checkpoint管理最佳实践清单

### 1.1 核心原则

| 原则 | 说明 | 优先级 |
|------|------|--------|
| **隔离性** | 不同stage的checkpoint必须物理隔离 | P0 |
| **可追溯性** | 每个checkpoint必须包含完整的元数据 | P0 |
| **防覆盖** | 必须有机制防止意外覆盖重要模型 | P0 |
| **可恢复性** | 支持从任意checkpoint断点续训 | P0 |
| **空间管理** | 定期清理旧checkpoint，保留关键版本 | P1 |

### 1.2 保存策略

```
✅ 推荐做法:
├── 定期保存: 每N个epoch保存一次
├── 最佳保存: 保存验证指标最好的模型
├── 最新保存: 始终保留最新的checkpoint用于恢复
├── 安全保存: 先保存到临时文件，再原子重命名
└── 多副本: 关键checkpoint异地备份

❌ 避免做法:
├── 只保留最后一个checkpoint
├── 直接覆盖正在使用的模型文件
├── 不记录任何训练元数据
└── 混合存放不同stage的模型
```

---

## 二、推荐目录结构

### 2.1 多Stage训练目录结构

```
experiments/
├── {project_name}/                          # 项目名称
│   ├── README.md                            # 实验说明文档
│   ├── config/                              # 配置文件
│   │   ├── stage1_pretrain.yaml
│   │   ├── stage2_finetune.yaml
│   │   └── stage3_distill.yaml
│   │
│   ├── checkpoints/                         # Checkpoint根目录
│   │   ├── stage1_pretrain/                 # Stage 1: 预训练
│   │   │   ├── best/                        # 最佳模型
│   │   │   │   ├── model_best_epoch_50.pth
│   │   │   │   └── model_best_epoch_100.pth
│   │   │   ├── latest/                      # 最新模型(用于恢复)
│   │   │   │   └── checkpoint_latest.pth
│   │   │   ├── periodic/                    # 定期保存
│   │   │   │   ├── checkpoint_epoch_010.pth
│   │   │   │   ├── checkpoint_epoch_020.pth
│   │   │   │   └── checkpoint_epoch_030.pth
│   │   │   └── interrupted/                 # 中断恢复点
│   │   │       └── checkpoint_interrupt_epoch_025.pth
│   │   │
│   │   ├── stage2_finetune/                 # Stage 2: 微调
│   │   │   ├── best/
│   │   │   ├── latest/
│   │   │   ├── periodic/
│   │   │   └── interrupted/
│   │   │
│   │   └── stage3_distill/                  # Stage 3: 蒸馏
│   │       ├── best/
│   │       ├── latest/
│   │       └── periodic/
│   │
│   ├── logs/                                # 训练日志
│   │   ├── stage1_pretrain/
│   │   │   ├── events.out.tfevents.*        # TensorBoard日志
│   │   │   └── train.log                    # 文本日志
│   │   ├── stage2_finetune/
│   │   └── stage3_distill/
│   │
│   ├── tensorboards/                        # TensorBoard可视化
│   │   ├── stage1_pretrain/
│   │   ├── stage2_finetune/
│   │   └── stage3_distill/
│   │
│   └── metadata/                            # 元数据存储
│       ├── stage1_pretrain_history.json
│       ├── stage2_finetune_history.json
│       └── training_manifest.json           # 训练清单
│
└── archived/                                # 归档目录
    └── {project_name}_20240101/             # 按日期归档
```

### 2.2 单Stage简化结构

```
experiments/
└── {project_name}/
    ├── config/
    ├── checkpoints/
    │   ├── best/              # 最佳模型
    │   ├── latest/            # 最新检查点
    │   └── periodic/          # 定期保存
    ├── logs/
    └── metadata/
```

---

## 三、命名约定规范

### 3.1 Checkpoint文件名格式

```python
# 标准命名格式
checkpoint_{type}_epoch_{epoch:04d}_step_{step:08d}_metric_{metric:.4f}.pth

# 各类型命名示例
checkpoint_best_epoch_0050_step_00125000_metric_0.9234.pth    # 最佳模型
checkpoint_periodic_epoch_0010_step_00025000.pth               # 定期保存
checkpoint_latest.pth                                          # 最新检查点(固定名)
checkpoint_interrupt_epoch_0025_step_00062500.pth              # 中断恢复点
```

### 3.2 命名组件说明

| 组件 | 格式 | 示例 | 说明 |
|------|------|------|------|
| prefix | `checkpoint` | checkpoint | 固定前缀 |
| type | `best/periodic/latest/interrupt` | best | checkpoint类型 |
| epoch | `epoch_{:04d}` | epoch_0050 | 训练轮次，4位补零 |
| step | `step_{:08d}` | step_00125000 | 全局步数，8位补零 |
| metric | `metric_{:.4f}` | metric_0.9234 | 关键指标值 |
| ext | `.pth/.pt/.safetensors` | .pth | 文件扩展名 |

### 3.3 版本标签规范

```python
# 语义化版本标签
v1.0.0-stage1-pretrain          # Stage 1完成版本
v1.1.0-stage1-resumed           # Stage 1恢复后版本
v2.0.0-stage2-finetune          # Stage 2完成版本
v2.1.0-stage2-best              # Stage 2最佳版本

# 在checkpoint中存储版本信息
checkpoint['version'] = 'v2.0.0-stage2-finetune'
checkpoint['parent_version'] = 'v1.0.0-stage1-pretrain'  # 父版本追溯
```

---

## 四、防止覆盖策略

### 4.1 多重保护机制

```python
# 策略1: 目录隔离
# 不同stage使用不同目录，物理隔离

# 策略2: 写入保护
import os

def protect_checkpoint(filepath):
    """设置文件只读保护"""
    if os.path.exists(filepath):
        # 设置为只读: 0o444 (r--r--r--)
        os.chmod(filepath, 0o444)

def unprotect_checkpoint(filepath):
    """解除只读保护"""
    if os.path.exists(filepath):
        # 设置为可写: 0o644 (rw-r--r--)
        os.chmod(filepath, 0o644)

# 策略3: 备份机制
def backup_before_overwrite(filepath, backup_dir):
    """覆盖前自动备份"""
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(filepath)
        backup_path = os.path.join(backup_dir, f'{filename}.{timestamp}.bak')
        shutil.copy2(filepath, backup_path)
        return backup_path
    return None

# 策略4: 版本控制集成
def tag_checkpoint_with_git(checkpoint_path, tag_name):
    """使用git标签标记重要checkpoint"""
    # 需要配合git-lfs管理大文件
    os.system(f'git add {checkpoint_path}')
    os.system(f'git commit -m "Add checkpoint: {tag_name}"')
    os.system(f'git tag {tag_name}')
```

### 4.2 原子写入操作

```python
import os
import tempfile
import shutil

def atomic_save_checkpoint(state_dict, filepath):
    """
    原子性保存checkpoint
    防止写入过程中断导致文件损坏
    """
    # 获取目标目录
    dir_name = os.path.dirname(filepath)
    
    # 在同目录创建临时文件
    fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        # 写入临时文件
        with os.fdopen(fd, 'wb') as f:
            torch.save(state_dict, f)
        
        # 原子重命名
        os.replace(temp_path, filepath)
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
```

### 4.3 Checkpoint锁定机制

```python
import fcntl
import errno

class CheckpointLock:
    """Checkpoint文件锁，防止并发写入"""
    
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.fd = None
    
    def acquire(self, blocking=True):
        """获取锁"""
        self.fd = open(self.lock_file, 'w')
        try:
            if blocking:
                fcntl.flock(self.fd, fcntl.LOCK_EX)
            else:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError as e:
            if e.errno == errno.EAGAIN:
                return False
            raise
    
    def release(self):
        """释放锁"""
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        self.release()
```

---

## 五、元数据记录规范

### 5.1 Checkpoint必须包含的元数据

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

### 5.2 元数据记录示例

```python
import torch
import json
from datetime import datetime
import platform
import subprocess

def get_git_commit():
    """获取当前git commit hash"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.getcwd()
        ).decode().strip()
    except:
        return 'unknown'

def create_checkpoint_metadata(model, optimizer, scheduler, config, 
                                metrics, epoch, global_step, stage_info):
    """创建完整的checkpoint元数据"""
    
    metadata = {
        # 训练状态
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
        # 训练配置
        'config': config,
        'stage': stage_info['name'],
        'stage_version': stage_info['version'],
        
        # 性能指标
        'metrics': metrics,
        'best_metric': stage_info.get('best_metric', None),
        
        # 系统信息
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'git_commit': get_git_commit(),
        
        # 版本控制
        'version': f"v{stage_info['version']}",
        'parent_checkpoint': stage_info.get('parent_checkpoint', None),
        'stage_history': stage_info.get('history', []),
    }
    
    return metadata
```

---

## 六、完整Python代码示例

### 6.1 CheckpointManager 完整实现

```python
"""
CheckpointManager - 深度学习Checkpoint管理器
支持多stage训练、断点续训、版本控制
"""

import os
import json
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, asdict
import platform
import subprocess

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class StageInfo:
    """Stage信息数据类"""
    name: str                      # stage名称
    version: str                   # stage版本
    parent_stage: Optional[str] = None  # 父stage
    parent_checkpoint: Optional[str] = None  # 父checkpoint路径
    best_metric: Optional[float] = None
    history: List[Dict] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class CheckpointConfig:
    """Checkpoint配置"""
    save_interval_epochs: int = 10      # 每N个epoch保存
    save_interval_steps: int = 10000    # 每N个step保存
    keep_last_n: int = 3                # 保留最近N个periodic checkpoint
    keep_best_n: int = 2                # 保留最佳N个checkpoint
    save_latest: bool = True            # 是否保存latest
    compress: bool = False              # 是否压缩


class CheckpointManager:
    """
    Checkpoint管理器
    
    特性:
    - 多stage隔离管理
    - 自动防止覆盖
    - 原子写入
    - 元数据完整记录
    - 自动清理旧checkpoint
    """
    
    def __init__(
        self,
        experiment_dir: str,
        stage_info: StageInfo,
        config: Optional[CheckpointConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.experiment_dir = Path(experiment_dir)
        self.stage_info = stage_info
        self.config = config or CheckpointConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建目录结构
        self._setup_directories()
        
        # 跟踪最佳指标
        self.best_metric = stage_info.best_metric or float('-inf')
        self.best_checkpoint_path = None
        
        # 加载历史记录
        self.checkpoint_history = self._load_history()
        
    def _setup_directories(self):
        """创建checkpoint目录结构"""
        stage_name = self.stage_info.name
        
        self.checkpoint_dir = self.experiment_dir / 'checkpoints' / stage_name
        self.best_dir = self.checkpoint_dir / 'best'
        self.latest_dir = self.checkpoint_dir / 'latest'
        self.periodic_dir = self.checkpoint_dir / 'periodic'
        self.interrupt_dir = self.checkpoint_dir / 'interrupted'
        self.metadata_dir = self.experiment_dir / 'metadata'
        
        # 创建所有目录
        for dir_path in [self.best_dir, self.latest_dir, 
                         self.periodic_dir, self.interrupt_dir, 
                         self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_history(self) -> List[Dict]:
        """加载checkpoint历史"""
        history_file = self.metadata_dir / f'{self.stage_info.name}_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """保存checkpoint历史"""
        history_file = self.metadata_dir / f'{self.stage_info.name}_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
    
    def _get_git_commit(self) -> str:
        """获取git commit hash"""
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=self.experiment_dir,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            return 'unknown'
    
    def _create_checkpoint_dict(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None
    ) -> Dict:
        """创建checkpoint字典"""
        
        checkpoint = {
            # 模型和优化器状态
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            
            # 训练进度
            'epoch': epoch,
            'global_step': global_step,
            
            # 指标
            'metrics': metrics,
            'best_metric': self.best_metric,
            
            # 配置
            'config': config or {},
            'stage': self.stage_info.name,
            'stage_version': self.stage_info.version,
            
            # 系统信息
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'git_commit': self._get_git_commit(),
            
            # 版本信息
            'version': f"{self.stage_info.version}-epoch{epoch}",
            'parent_checkpoint': self.stage_info.parent_checkpoint,
            'stage_history': self.stage_info.history,
        }
        
        return checkpoint
    
    def _atomic_save(self, checkpoint: Dict, filepath: Path):
        """原子性保存checkpoint"""
        temp_path = filepath.parent / f'.tmp_{filepath.name}'
        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, filepath)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _cleanup_old_checkpoints(self, keep_n: int, directory: Path):
        """清理旧的checkpoint"""
        checkpoints = sorted(
            directory.glob('checkpoint_*.pth'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[keep_n:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {old_checkpoint}: {e}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        is_best: bool = False,
        is_interrupt: bool = False
    ) -> Dict[str, Path]:
        """
        保存checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            global_step: 全局步数
            metrics: 指标字典
            config: 配置字典
            is_best: 是否为最佳模型
            is_interrupt: 是否为中断保存
            
        Returns:
            保存的checkpoint路径字典
        """
        checkpoint = self._create_checkpoint_dict(
            model, optimizer, scheduler, epoch, global_step, metrics, config
        )
        
        saved_paths = {}
        
        # 1. 保存latest checkpoint
        if self.config.save_latest:
            latest_path = self.latest_dir / 'checkpoint_latest.pth'
            self._atomic_save(checkpoint, latest_path)
            saved_paths['latest'] = latest_path
            self.logger.info(f"Saved latest checkpoint: {latest_path}")
        
        # 2. 保存periodic checkpoint
        if epoch % self.config.save_interval_epochs == 0:
            metric_str = f"_metric_{metrics.get('val_loss', 0):.4f}" if 'val_loss' in metrics else ""
            periodic_name = f"checkpoint_epoch_{epoch:04d}_step_{global_step:08d}{metric_str}.pth"
            periodic_path = self.periodic_dir / periodic_name
            self._atomic_save(checkpoint, periodic_path)
            saved_paths['periodic'] = periodic_path
            self.logger.info(f"Saved periodic checkpoint: {periodic_path}")
            
            # 清理旧checkpoint
            self._cleanup_old_checkpoints(self.config.keep_last_n, self.periodic_dir)
        
        # 3. 保存best checkpoint
        current_metric = metrics.get('val_accuracy', metrics.get('val_loss', 0))
        metric_name = 'val_accuracy' if 'val_accuracy' in metrics else 'val_loss'
        
        # 根据指标类型判断是否为最佳
        is_better = False
        if metric_name == 'val_accuracy':
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric if self.best_metric != float('-inf') else True
        
        if is_best or is_better:
            self.best_metric = current_metric
            checkpoint['best_metric'] = self.best_metric
            
            best_name = f"checkpoint_best_epoch_{epoch:04d}_step_{global_step:08d}_metric_{current_metric:.4f}.pth"
            best_path = self.best_dir / best_name
            self._atomic_save(checkpoint, best_path)
            saved_paths['best'] = best_path
            self.best_checkpoint_path = best_path
            self.logger.info(f"Saved best checkpoint: {best_path}")
            
            # 清理旧best checkpoint
            self._cleanup_old_checkpoints(self.config.keep_best_n, self.best_dir)
        
        # 4. 保存interrupt checkpoint
        if is_interrupt:
            interrupt_name = f"checkpoint_interrupt_epoch_{epoch:04d}_step_{global_step:08d}.pth"
            interrupt_path = self.interrupt_dir / interrupt_name
            self._atomic_save(checkpoint, interrupt_path)
            saved_paths['interrupt'] = interrupt_path
            self.logger.info(f"Saved interrupt checkpoint: {interrupt_path}")
        
        # 更新历史记录
        self.checkpoint_history.append({
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'paths': {k: str(v) for k, v in saved_paths.items()}
        })
        self._save_history()
        
        return saved_paths
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        load_type: str = 'latest'
    ) -> Dict:
        """
        加载checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器(可选)
            scheduler: 学习率调度器(可选)
            checkpoint_path: 指定checkpoint路径
            load_type: 加载类型 ('latest', 'best', 'interrupt')
            
        Returns:
            checkpoint字典
        """
        if checkpoint_path is None:
            # 根据load_type自动选择
            if load_type == 'latest':
                checkpoint_path = self.latest_dir / 'checkpoint_latest.pth'
            elif load_type == 'best' and self.best_checkpoint_path:
                checkpoint_path = self.best_checkpoint_path
            elif load_type == 'interrupt':
                # 找最新的interrupt checkpoint
                interrupt_checkpoints = list(self.interrupt_dir.glob('checkpoint_interrupt_*.pth'))
                if interrupt_checkpoints:
                    checkpoint_path = max(interrupt_checkpoints, key=lambda x: x.stat().st_mtime)
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复最佳指标
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
                        f"step {checkpoint['global_step']}")
        
        return checkpoint
    
    def get_stage_transition_info(self) -> Dict:
        """获取stage转换信息"""
        return {
            'stage_name': self.stage_info.name,
            'stage_version': self.stage_info.version,
            'best_checkpoint': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            'best_metric': self.best_metric,
            'total_epochs': self.checkpoint_history[-1]['epoch'] if self.checkpoint_history else 0,
            'total_steps': self.checkpoint_history[-1]['global_step'] if self.checkpoint_history else 0,
        }


class MultiStageTrainingManager:
    """多stage训练管理器"""
    
    def __init__(self, experiment_dir: str, project_name: str):
        self.experiment_dir = Path(experiment_dir) / project_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages: List[StageInfo] = []
        self.current_stage_idx = -1
        self.checkpoint_manager: Optional[CheckpointManager] = None
        
        # 加载已有stage信息
        self._load_manifest()
    
    def _load_manifest(self):
        """加载训练清单"""
        manifest_path = self.experiment_dir / 'metadata' / 'training_manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                data = json.load(f)
                self.stages = [StageInfo(**s) for s in data.get('stages', [])]
    
    def _save_manifest(self):
        """保存训练清单"""
        manifest_path = self.experiment_dir / 'metadata' / 'training_manifest.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'project_name': self.experiment_dir.name,
            'stages': [asdict(s) for s in self.stages],
            'current_stage': self.current_stage_idx,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def start_stage(
        self,
        stage_name: str,
        stage_version: str,
        parent_stage: Optional[str] = None,
        parent_checkpoint: Optional[str] = None,
        checkpoint_config: Optional[CheckpointConfig] = None
    ) -> CheckpointManager:
        """
        开始新的训练stage
        
        Args:
            stage_name: stage名称
            stage_version: stage版本
            parent_stage: 父stage名称
            parent_checkpoint: 父stage的checkpoint路径
            checkpoint_config: checkpoint配置
            
        Returns:
            CheckpointManager实例
        """
        # 检查stage是否已存在
        for stage in self.stages:
            if stage.name == stage_name:
                raise ValueError(f"Stage '{stage_name}' already exists!")
        
        # 创建stage信息
        stage_info = StageInfo(
            name=stage_name,
            version=stage_version,
            parent_stage=parent_stage,
            parent_checkpoint=parent_checkpoint,
            history=[s.name for s in self.stages]
        )
        
        self.stages.append(stage_info)
        self.current_stage_idx = len(self.stages) - 1
        
        # 创建CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            experiment_dir=self.experiment_dir,
            stage_info=stage_info,
            config=checkpoint_config
        )
        
        self._save_manifest()
        
        return self.checkpoint_manager
    
    def transition_to_next_stage(
        self,
        next_stage_name: str,
        next_stage_version: str,
        checkpoint_config: Optional[CheckpointConfig] = None
    ) -> CheckpointManager:
        """
        转换到下一个stage
        
        Args:
            next_stage_name: 下一个stage名称
            next_stage_version: 下一个stage版本
            checkpoint_config: checkpoint配置
            
        Returns:
            新的CheckpointManager实例
        """
        if self.checkpoint_manager is None:
            raise RuntimeError("No active stage. Call start_stage() first.")
        
        # 获取当前stage的最佳checkpoint
        transition_info = self.checkpoint_manager.get_stage_transition_info()
        parent_checkpoint = transition_info['best_checkpoint']
        
        current_stage_name = self.stages[self.current_stage_idx].name
        
        # 开始新stage
        return self.start_stage(
            stage_name=next_stage_name,
            stage_version=next_stage_version,
            parent_stage=current_stage_name,
            parent_checkpoint=parent_checkpoint,
            checkpoint_config=checkpoint_config
        )
    
    def list_stages(self) -> List[Dict]:
        """列出所有stage"""
        return [
            {
                'index': i,
                'name': s.name,
                'version': s.version,
                'parent_stage': s.parent_stage,
            }
            for i, s in enumerate(self.stages)
        ]
```

### 6.2 使用示例

```python
"""多stage训练完整示例"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 假设的模型和数据
def create_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def get_dataloaders():
    # 返回训练集和验证集的DataLoader
    pass

def train_epoch(model, dataloader, optimizer, device):
    # 训练一个epoch
    pass

def validate(model, dataloader, device):
    # 验证，返回指标
    return {'val_loss': 0.5, 'val_accuracy': 0.85}


def main():
    # 配置
    experiment_dir = './experiments'
    project_name = 'my_project'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建多stage训练管理器
    stage_manager = MultiStageTrainingManager(experiment_dir, project_name)
    
    # ============ Stage 1: 预训练 ============
    print("=" * 50)
    print("Starting Stage 1: Pre-training")
    print("=" * 50)
    
    checkpoint_config = CheckpointConfig(
        save_interval_epochs=5,
        keep_last_n=3,
        keep_best_n=2
    )
    
    ckpt_manager = stage_manager.start_stage(
        stage_name='stage1_pretrain',
        stage_version='1.0.0',
        checkpoint_config=checkpoint_config
    )
    
    # 创建模型和优化器
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 尝试恢复训练
    checkpoint = ckpt_manager.load_checkpoint(model, optimizer, scheduler, load_type='latest')
    start_epoch = checkpoint['epoch'] + 1 if checkpoint else 0
    global_step = checkpoint['global_step'] if checkpoint else 0
    
    # Stage 1训练循环
    train_loader, val_loader = get_dataloaders()
    num_epochs = 50
    
    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_metrics = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 保存checkpoint
        metrics = {**train_metrics, **val_metrics}
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config={'lr': 1e-3, 'batch_size': 32},
            is_best=False  # 自动判断是否为最佳
        )
        
        global_step += len(train_loader)
    
    # ============ Stage 2: 微调 ============
    print("=" * 50)
    print("Starting Stage 2: Fine-tuning")
    print("=" * 50)
    
    # 转换到下一个stage
    ckpt_manager = stage_manager.transition_to_next_stage(
        next_stage_name='stage2_finetune',
        next_stage_version='2.0.0',
        checkpoint_config=CheckpointConfig(save_interval_epochs=2)
    )
    
    # 加载上一个stage的最佳模型
    checkpoint = ckpt_manager.load_checkpoint(model, optimizer, scheduler, load_type='best')
    
    # 修改优化器用于微调
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Stage 2训练循环
    for epoch in range(20):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        
        metrics = {**train_metrics, **val_metrics}
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics
        )
        
        global_step += len(train_loader)
    
    # 打印所有stage
    print("\nAll stages:")
    for stage_info in stage_manager.list_stages():
        print(f"  {stage_info}")


if __name__ == '__main__':
    main()
```

---

## 七、快速参考卡片

### 7.1 目录结构速查

```
checkpoints/
├── {stage_name}/
│   ├── best/           # 最佳模型 (保留N个)
│   ├── latest/         # 最新检查点 (1个)
│   ├── periodic/       # 定期保存 (保留N个)
│   └── interrupted/    # 中断恢复点
```

### 7.2 命名格式速查

```
checkpoint_{type}_epoch_{EEEE}_step_{SSSSSSSS}_metric_{M.MMMM}.pth
```

### 7.3 关键API速查

```python
# 创建管理器
manager = MultiStageTrainingManager('./exp', 'project')

# 开始stage
ckpt_mgr = manager.start_stage('stage1', '1.0.0')

# 保存checkpoint
ckpt_mgr.save_checkpoint(model, optimizer, scheduler, epoch, step, metrics)

# 加载checkpoint
checkpoint = ckpt_mgr.load_checkpoint(model, optimizer, scheduler, load_type='latest')

# 转换stage
ckpt_mgr = manager.transition_to_next_stage('stage2', '2.0.0')
```

---

## 八、常见问题FAQ

**Q1: 如何避免Stage 2覆盖Stage 1的模型？**
> A: 使用`MultiStageTrainingManager`，每个stage有独立的目录，物理隔离。

**Q2: 如何回到某个stage重新训练？**
> A: 使用`load_checkpoint`指定stage的checkpoint路径，或重新`start_stage`并指定`parent_checkpoint`。

**Q3: 如何防止训练中断导致checkpoint损坏？**
> A: 使用`_atomic_save`方法，先写入临时文件再原子重命名。

**Q4: 如何管理磁盘空间？**
> A: 配置`keep_last_n`和`keep_best_n`参数，自动清理旧checkpoint。

**Q5: 如何记录完整的训练历史？**
> A: 元数据自动保存在checkpoint中，历史记录保存在`metadata/{stage}_history.json`。

---

*文档版本: 1.0 | 最后更新: 2024*
