# 计算机视觉与医学图像深度学习训练规范清单

> **适用领域**: 计算机视觉(CV) / 医学图像分析  
> **版本**: v1.0  
> **最后更新**: 2024

---

## 目录
1. [CV/医学图像训练特有注意事项](#1-cv医学图像训练特有注意事项)
2. [数据预处理与增强最佳实践](#2-数据预处理与增强最佳实践)
3. [领域特定评估指标](#3-领域特定评估指标)
4. [数据加载与内存管理优化](#4-数据加载与内存管理优化)
5. [多模态医学图像处理](#5-多模态医学图像处理)
6. [医学图像伦理与合规考虑](#6-医学图像伦理与合规考虑)
7. [完整代码示例库](#7-完整代码示例库)

---

## 1. CV/医学图像训练特有注意事项

### 1.1 医学图像特有挑战

| 挑战类型 | 具体描述 | 应对策略 |
|---------|---------|---------|
| **类别不平衡** | 病灶区域通常只占图像的1-10% | 使用加权损失、Focal Loss、重采样 |
| **小样本问题** | 标注数据稀缺且获取成本高 | 数据增强、迁移学习、半监督学习 |
| **多尺度特征** | 病灶大小差异巨大 | 多尺度训练、FPN、U-Net++ |
| **边界模糊** | 病灶边界不清晰 | 边界感知损失、注意力机制 |
| **噪声干扰** | 医学图像存在设备噪声 | 去噪预处理、鲁棒损失函数 |

### 1.2 CV通用注意事项

```python
# 关键配置检查清单
TRAINING_CHECKLIST = {
    # 数据层面
    "data_validation": [
        "检查图像尺寸一致性",
        "验证标签与图像对齐",
        "检查异常值和损坏文件",
        "确认数据分布（训练/验证/测试）"
    ],

    # 训练层面
    "training_setup": [
        "设置确定性训练（复现性）",
        "选择适当的初始化方法",
        "配置学习率调度策略",
        "设置早停机制"
    ],

    # 评估层面
    "evaluation": [
        "使用交叉验证",
        "报告置信区间",
        "进行统计显著性检验",
        "分析失败案例"
    ]
}
```

### 1.3 确定性训练设置（确保可复现性）

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """设置全局随机种子确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置Python哈希种子
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[INFO] 随机种子已设置为: {seed}")

# 使用示例
set_seed(42)
```

---

## 2. 数据预处理与增强最佳实践

### 2.1 医学图像预处理规范

```python
import torch
import numpy as np
from typing import Tuple, Optional
import SimpleITK as sitk

class MedicalImagePreprocessor:
    """医学图像预处理器"""

    def __init__(self, 
                 target_spacing: Optional[Tuple[float, ...]] = None,
                 window_center: Optional[float] = None,
                 window_width: Optional[float] = None,
                 normalize_method: str = 'zscore'):
        """
        Args:
            target_spacing: 目标体素间距 (z, y, x)
            window_center: 窗位 (CT图像)
            window_width: 窗宽 (CT图像)
            normalize_method: 归一化方法 ('zscore', 'minmax', 'window')
        """
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.normalize_method = normalize_method

    def apply_window(self, image: np.ndarray) -> np.ndarray:
        """应用CT窗宽窗位"""
        if self.window_center is None or self.window_width is None:
            return image

        min_val = self.window_center - self.window_width // 2
        max_val = self.window_center + self.window_width // 2

        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        return windowed.astype(np.float32)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化"""
        if self.normalize_method == 'zscore':
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-8)

        elif self.normalize_method == 'minmax':
            min_val = np.min(image)
            max_val = np.max(image)
            return (image - min_val) / (max_val - min_val + 1e-8)

        elif self.normalize_method == 'window':
            return self.apply_window(image)

        else:
            raise ValueError(f"未知的归一化方法: {self.normalize_method}")

    def resample(self, image: sitk.Image, label: Optional[sitk.Image] = None):
        """重采样到目标间距"""
        if self.target_spacing is None:
            return image, label

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # 计算新尺寸
        new_size = [
            int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(len(original_spacing))
        ]

        # 图像使用线性插值
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image = resampler.Execute(image)

        # 标签使用最近邻插值
        if label is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_label = resampler.Execute(label)
            return resampled_image, resampled_label

        return resampled_image, None
```

### 2.2 数据增强策略

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

class MedicalAugmentation:
    """医学图像数据增强"""

    @staticmethod
    def get_training_augmentation_2d(image_size: Tuple[int, int] = (512, 512)):
        """2D医学图像训练增强"""
        return A.Compose([
            # 几何变换
            A.RandomResizedCrop(
                height=image_size[0], 
                width=image_size[1], 
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),

            # 弹性形变（医学图像特有）
            A.ElasticTransform(
                alpha=1, 
                sigma=50, 
                alpha_affine=50,
                p=0.3
            ),

            # 强度变换
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # 噪声（模拟设备噪声）
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # 模糊
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # 归一化
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])

    @staticmethod
    def get_training_augmentation_3d():
        """3D医学图像训练增强配置"""
        # 3D增强需要使用专门的库如torchio或monai
        return {
            "random_flip": {"axes": [0, 1, 2], "p": 0.5},
            "random_affine": {
                "scales": (0.9, 1.1),
                "degrees": 15,
                "p": 0.3
            },
            "random_elastic_deformation": {
                "num_control_points": 7,
                "max_displacement": 7.5,
                "p": 0.2
            },
            "random_noise": {"mean": 0, "std": 0.1, "p": 0.2},
            "random_gamma": {"log_gamma": (-0.3, 0.3), "p": 0.3}
        }

    @staticmethod
    def get_validation_augmentation_2d():
        """验证/测试时增强（仅归一化）"""
        return A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])
```

### 2.3 类别不平衡处理

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C, H, W] 或 [N, C, D, H, W]
            targets: [N, H, W] 或 [N, D, H, W]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss - 医学图像分割常用"""

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, C, H, W] - softmax输出
            targets: [N, H, W] 或 [N, C, H, W] - one-hot编码
        """
        # 将targets转为one-hot
        if targets.dim() == inputs.dim() - 1:
            targets = F.one_hot(targets, num_classes=inputs.shape[1])
            targets = targets.permute(0, -1, *range(1, targets.dim()-1)).float()

        # Softmax
        inputs = F.softmax(inputs, dim=1)

        # 展平
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        # 计算Dice
        intersection = (inputs * targets).sum(dim=2)
        union = inputs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """组合损失：Dice + CE"""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def compute_class_weights(labels: torch.Tensor, num_classes: int, method: str = 'inverse'):
    """计算类别权重"""
    class_counts = torch.bincount(labels.flatten(), minlength=num_classes).float()

    if method == 'inverse':
        weights = 1.0 / (class_counts + 1e-8)
    elif method == 'inverse_sqrt':
        weights = 1.0 / torch.sqrt(class_counts + 1e-8)
    elif method == 'effective':
        # 有效样本数
        beta = 0.9999
        effective_num = (1.0 - beta) / (1.0 - beta ** class_counts)
        weights = 1.0 / effective_num
    else:
        raise ValueError(f"未知方法: {method}")

    # 归一化
    weights = weights / weights.sum() * num_classes
    return weights
```

---

## 3. 领域特定评估指标

### 3.1 分割指标实现

```python
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, List, Tuple, Optional

class SegmentationMetrics:
    """医学图像分割评估指标"""

    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """
        Dice系数 (F1-Score)

        Args:
            pred: 预测掩码 [H, W] 或 [D, H, W]
            target: 真实掩码 [H, W] 或 [D, H, W]
        """
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)

    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """
        IoU (Jaccard Index)

        Args:
            pred: 预测掩码
            target: 真实掩码
        """
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)

    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray, percentile: float = 95.0) -> float:
        """
        Hausdorff Distance (HD) 和 HD95

        Args:
            pred: 预测掩码 (二值)
            target: 真实掩码 (二值)
            percentile: 百分位数 (95表示HD95)
        """
        # 获取边界点
        pred_points = np.argwhere(pred > 0)
        target_points = np.argwhere(target > 0)

        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')

        # 计算有向Hausdorff距离
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]

        if percentile == 100.0:
            return max(d1, d2)

        # 计算HD95
        from scipy.spatial import cKDTree

        tree1 = cKDTree(pred_points)
        tree2 = cKDTree(target_points)

        distances1, _ = tree1.query(target_points)
        distances2, _ = tree2.query(pred_points)

        all_distances = np.concatenate([distances1, distances2])
        hd95 = np.percentile(all_distances, percentile)

        return float(hd95)

    @staticmethod
    def sensitivity_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """敏感度/召回率 (Recall)"""
        tp = np.sum(pred * target)
        fn = np.sum((1 - pred) * target)
        return float((tp + smooth) / (tp + fn + smooth))

    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """特异度"""
        tn = np.sum((1 - pred) * (1 - target))
        fp = np.sum(pred * (1 - target))
        return float((tn + smooth) / (tn + fp + smooth))

    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """精确率"""
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        return float((tp + smooth) / (tp + fp + smooth))

    @staticmethod
    def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
        """准确率"""
        return float(np.mean(pred == target))

    @staticmethod
    def compute_all_metrics(pred: np.ndarray, target: np.ndarray, 
                           include_hd: bool = True) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {
            'dice': SegmentationMetrics.dice_coefficient(pred, target),
            'iou': SegmentationMetrics.iou_score(pred, target),
            'sensitivity': SegmentationMetrics.sensitivity_recall(pred, target),
            'specificity': SegmentationMetrics.specificity(pred, target),
            'precision': SegmentationMetrics.precision(pred, target),
            'accuracy': SegmentationMetrics.accuracy(pred, target),
        }

        if include_hd:
            metrics['hd95'] = SegmentationMetrics.hausdorff_distance(pred, target, 95.0)
            metrics['hd100'] = SegmentationMetrics.hausdorff_distance(pred, target, 100.0)

        return metrics
```

### 3.2 分类与检测指标

```python
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt

class ClassificationMetrics:
    """分类任务评估指标"""

    @staticmethod
    def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray, 
                         multi_class: str = 'ovr') -> float:
        """
        计算AUC-ROC

        Args:
            y_true: 真实标签 [N]
            y_score: 预测概率 [N, C]
            multi_class: 'ovr' (One-vs-Rest) 或 'ovo' (One-vs-One)
        """
        if y_score.ndim == 1 or y_score.shape[1] == 2:
            # 二分类
            return roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
        else:
            # 多分类
            return roc_auc_score(y_true, y_score, multi_class=multi_class, average='macro')

    @staticmethod
    def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算AUC-PR (Average Precision)"""
        if y_score.ndim == 1 or y_score.shape[1] == 2:
            return average_precision_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
        else:
            return average_precision_score(y_true, y_score, average='macro')

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                       class_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))

        if y_score.ndim == 1 or y_score.shape[1] == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
            auc = roc_auc_score(y_true, y_score[:, 1] if y_score.ndim > 1 else y_score)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        else:
            # 多分类 - One-vs-Rest
            n_classes = y_score.shape[1]
            y_true_bin = np.eye(n_classes)[y_true]

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
                name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

---

## 4. 数据加载与内存管理优化

### 4.1 高效DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import nibabel as nib
from typing import Callable, Optional, List, Dict

class MedicalImageDataset(Dataset):
    """高效医学图像数据集"""

    def __init__(self,
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 cache_data: bool = False,
                 preload: bool = False):
        """
        Args:
            data_dir: 数据目录
            transform: 数据变换
            cache_data: 是否缓存数据
            preload: 是否预加载所有数据
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}

        # 获取所有样本
        self.samples = self._get_samples()

        # 预加载
        if preload:
            self._preload_data()

    def _get_samples(self) -> List[Dict]:
        """获取所有样本路径"""
        samples = []
        image_dir = self.data_dir / 'images'
        label_dir = self.data_dir / 'labels'

        for img_path in sorted(image_dir.glob('*.nii.gz')):
            label_path = label_dir / img_path.name
            if label_path.exists():
                samples.append({
                    'image': str(img_path),
                    'label': str(label_path),
                    'id': img_path.stem.replace('.nii', '')
                })

        return samples

    def _preload_data(self):
        """预加载所有数据到内存"""
        print(f"[INFO] 预加载 {len(self.samples)} 个样本...")
        for i, sample in enumerate(self.samples):
            self.cache[i] = self._load_sample(sample)
            if (i + 1) % 10 == 0:
                print(f"[INFO] 已加载 {i + 1}/{len(self.samples)}")
        print("[INFO] 预加载完成")

    def _load_sample(self, sample: Dict) -> Dict:
        """加载单个样本"""
        # 使用nibabel加载
        image = nib.load(sample['image']).get_fdata()
        label = nib.load(sample['label']).get_fdata()

        return {
            'image': image.astype(np.float32),
            'label': label.astype(np.int64),
            'id': sample['id']
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        # 检查缓存
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            sample = self._load_sample(self.samples[idx])
            if self.cache_data:
                self.cache[idx] = sample

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_optimized_dataloader(dataset: Dataset,
                              batch_size: int = 4,
                              num_workers: int = 4,
                              pin_memory: bool = True) -> DataLoader:
    """
    创建优化的DataLoader

    优化策略：
    1. 多进程数据加载
    2. 固定内存(pin_memory)加速GPU传输
    3. 预取数据
    4. 自动批处理
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,  # 保持worker进程
        prefetch_factor=2 if num_workers > 0 else None,
    )
```

### 4.2 3D数据补丁采样

```python
class PatchSampler:
    """3D图像补丁采样器"""

    def __init__(self, patch_size: Tuple[int, int, int] = (128, 128, 128),
                 stride: Optional[Tuple[int, int, int]] = None,
                 mode: str = 'random'):
        """
        Args:
            patch_size: 补丁大小 (D, H, W)
            stride: 滑动窗口步长
            mode: 'random' 或 'sliding'
        """
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.mode = mode

    def random_sample(self, image: np.ndarray, label: np.ndarray,
                      num_samples: int = 1, foreground_ratio: float = 0.5) -> List[Dict]:
        """
        随机采样补丁

        Args:
            image: [C, D, H, W]
            label: [D, H, W]
            num_samples: 采样数量
            foreground_ratio: 前景区域采样比例
        """
        patches = []
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size

        for _ in range(num_samples):
            # 决定是否采样前景区域
            if np.random.rand() < foreground_ratio and label.max() > 0:
                # 在前景区域采样
                foreground_coords = np.argwhere(label > 0)
                if len(foreground_coords) > 0:
                    center = foreground_coords[np.random.randint(len(foreground_coords))]
                else:
                    center = [np.random.randint(pd//2, d-pd//2),
                             np.random.randint(ph//2, h-ph//2),
                             np.random.randint(pw//2, w-pw//2)]
            else:
                # 随机采样
                center = [np.random.randint(pd//2, d-pd//2),
                         np.random.randint(ph//2, h-ph//2),
                         np.random.randint(pw//2, w-pw//2)]

            # 提取补丁
            d_start = center[0] - pd // 2
            h_start = center[1] - ph // 2
            w_start = center[2] - pw // 2

            image_patch = image[:, d_start:d_start+pd, 
                               h_start:h_start+ph, 
                               w_start:w_start+pw]
            label_patch = label[d_start:d_start+pd,
                               h_start:h_start+ph,
                               w_start:w_start+pw]

            patches.append({
                'image': image_patch,
                'label': label_patch,
                'center': center
            })

        return patches
```

### 4.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """混合精度训练器"""

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()  # 梯度缩放器

    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        # 自动混合精度
        with autocast():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        # 反向传播与梯度缩放
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()
```

---

## 5. 多模态医学图像处理

### 5.1 多模态数据加载

```python
class MultimodalMedicalDataset(Dataset):
    """多模态医学图像数据集"""

    def __init__(self,
                 data_dir: str,
                 modalities: List[str] = ['T1', 'T2', 'FLAIR', 'T1ce'],
                 transform: Optional[Callable] = None):
        """
        Args:
            data_dir: 数据目录
            modalities: 模态列表 (如MRI的T1, T2, FLAIR等)
            transform: 数据变换
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.transform = transform
        self.samples = self._get_samples()

    def _get_samples(self) -> List[Dict]:
        """获取样本列表"""
        samples = []

        # 假设目录结构: data_dir/subject_id/modality.nii.gz
        for subject_dir in sorted(self.data_dir.iterdir()):
            if subject_dir.is_dir():
                sample = {'id': subject_dir.name}

                # 检查所有模态是否存在
                has_all_modalities = True
                for mod in self.modalities:
                    mod_path = subject_dir / f'{mod}.nii.gz'
                    if mod_path.exists():
                        sample[mod] = str(mod_path)
                    else:
                        has_all_modalities = False
                        break

                # 检查标签
                label_path = subject_dir / 'label.nii.gz'
                if label_path.exists():
                    sample['label'] = str(label_path)

                if has_all_modalities:
                    samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载所有模态
        images = []
        for mod in self.modalities:
            img = nib.load(sample[mod]).get_fdata()
            images.append(img.astype(np.float32))

        # 堆叠为 [C, D, H, W]
        image = np.stack(images, axis=0)

        # 加载标签
        label = nib.load(sample['label']).get_fdata().astype(np.int64)

        data = {
            'image': image,
            'label': label,
            'id': sample['id']
        }

        if self.transform:
            data = self.transform(data)

        return data


# 模态融合模块
class ModalityFusion(nn.Module):
    """多模态融合模块"""

    def __init__(self, num_modalities: int, channels: int, fusion_type: str = 'concat'):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 拼接后1x1卷积降维
            self.fusion_conv = nn.Conv3d(channels * num_modalities, channels, 1)
        elif fusion_type == 'attention':
            # 模态注意力
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels * num_modalities, num_modalities, 1),
                nn.Sigmoid()
            )
        elif fusion_type == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Conv3d(channels * num_modalities, num_modalities, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            modalities: 各模态特征列表 [B, C, D, H, W]

        Returns:
            融合特征 [B, C, D, H, W]
        """
        if self.fusion_type == 'concat':
            x = torch.cat(modalities, dim=1)
            return self.fusion_conv(x)

        elif self.fusion_type == 'attention':
            x = torch.cat(modalities, dim=1)
            attn = self.attention(x)  # [B, num_mod, 1, 1, 1]

            # 加权融合
            fused = sum(m * attn[:, i:i+1] for i, m in enumerate(modalities))
            return fused

        elif self.fusion_type == 'gated':
            x = torch.cat(modalities, dim=1)
            gates = self.gate(x)  # [B, num_mod, D, H, W]

            # 门控加权
            fused = sum(m * gates[:, i:i+1] for i, m in enumerate(modalities))
            return fused

        else:
            raise ValueError(f"未知融合类型: {self.fusion_type}")
```

---

## 6. 医学图像伦理与合规考虑

### 6.1 数据隐私保护

```python
import hashlib
import json
from datetime import datetime
from typing import Dict, Any

class MedicalDataPrivacy:
    """医学数据隐私保护工具"""

    @staticmethod
    def pseudonymize_patient_id(patient_id: str, salt: str = '') -> str:
        """
        患者ID去标识化

        使用哈希函数生成假名
        """
        hash_input = f"{patient_id}{salt}".encode('utf-8')
        pseudonym = hashlib.sha256(hash_input).hexdigest()[:16]
        return pseudonym

    @staticmethod
    def remove_phi_metadata(image_path: str) -> Dict[str, Any]:
        """
        移除DICOM中的PHI (Protected Health Information)

        需要保留的字段:
        - 图像像素数据
        - 图像尺寸、间距等几何信息

        需要移除/匿名化的字段:
        - 患者姓名、ID
        - 出生日期
        - 医院信息
        - 检查日期（可保留年份用于研究）
        """
        try:
            import pydicom

            ds = pydicom.dcmread(image_path)

            # 匿名化标签
            phi_tags = [
                (0x0010, 0x0010),  # PatientName
                (0x0010, 0x0020),  # PatientID
                (0x0010, 0x0030),  # PatientBirthDate
                (0x0010, 0x0040),  # PatientSex
                (0x0008, 0x0090),  # ReferringPhysicianName
                (0x0008, 0x1050),  # PerformingPhysicianName
                (0x0008, 0x1070),  # OperatorsName
                (0x0010, 0x1040),  # PatientAddress
                (0x0010, 0x2154),  # PatientTelephoneNumbers
            ]

            for tag in phi_tags:
                if tag in ds:
                    # 保留年份用于研究
                    if tag == (0x0010, 0x0030):
                        birth_year = ds[tag].value[:4]
                        ds[tag].value = f"{birth_year}0101"
                    else:
                        ds[tag].value = "ANONYMIZED"

            return {'status': 'success', 'anonymized_fields': len(phi_tags)}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @staticmethod
    def log_data_access(user_id: str, action: str, 
                        patient_pseudonym: str, 
                        purpose: str) -> Dict[str, Any]:
        """
        记录数据访问日志（审计追踪）

        符合HIPAA/GDPR要求
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': hashlib.sha256(user_id.encode()).hexdigest()[:16],
            'action': action,
            'patient_pseudonym': patient_pseudonym,
            'purpose': purpose,
            'access_granted': True
        }

        # 保存日志（实际应用中应写入安全的数据库）
        with open('data_access_log.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return log_entry
```

### 6.2 合规检查清单

```python
MEDICAL_AI_COMPLIANCE_CHECKLIST = {
    "数据层面": {
        "数据获取": [
            "获得IRB（伦理委员会）批准",
            "获取患者知情同意",
            "数据使用协议(DUA)签署",
            "确认数据去标识化",
        ],
        "数据存储": [
            "使用加密存储",
            "实施访问控制",
            "定期备份",
            "制定数据保留政策",
        ],
        "数据传输": [
            "使用安全传输协议",
            "端到端加密",
            "审计日志记录",
        ]
    },

    "模型层面": {
        "模型训练": [
            "使用代表性数据集",
            "报告数据集偏差",
            "实施公平性评估",
            "记录训练配置",
        ],
        "模型验证": [
            "多中心外部验证",
            "前瞻性验证",
            "与金标准对比",
            "报告置信区间",
        ],
        "模型部署": [
            "性能监控",
            "漂移检测",
            "人工审核机制",
            "错误报告系统",
        ]
    },

    "法规层面": {
        "FDA (美国)": [
            "确定设备分类",
            "510(k) 或 PMA 申请",
            "临床试验数据",
            "质量管理体系(QSR)",
        ],
        "CE (欧洲)": [
            "MDR/IVDR合规",
            "公告机构审核",
            "临床证据",
            "上市后监督",
        ],
        "NMPA (中国)": [
            "医疗器械注册",
            "临床试验审批",
            "生产质量管理规范",
            "不良事件监测",
        ]
    }
}


def print_compliance_checklist():
    """打印合规检查清单"""
    for category, subcategories in MEDICAL_AI_COMPLIANCE_CHECKLIST.items():
        print(f"\n{'='*50}")
        print(f"📋 {category}")
        print('='*50)

        for subcategory, items in subcategories.items():
            print(f"\n  📁 {subcategory}")
            for item in items:
                print(f"    ✓ {item}")
```

---

## 7. 完整代码示例库

### 7.1 完整训练流程

```python
"""
医学图像分割完整训练流程示例
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

class MedicalImageTrainer:
    """医学图像训练器"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: dict):
        """
        Args:
            model: 网络模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = self._get_criterion()

        # 优化器
        self.optimizer = self._get_optimizer()

        # 学习率调度
        self.scheduler = self._get_scheduler()

        # 混合精度
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 日志
        self.writer = SummaryWriter(config.get('log_dir', 'runs'))
        self.best_dice = 0.0
        self.epoch = 0

        # 创建保存目录
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_criterion(self):
        """获取损失函数"""
        loss_type = self.config.get('loss', 'combined')

        if loss_type == 'dice':
            return DiceLoss()
        elif loss_type == 'ce':
            return nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            return FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'combined':
            return CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        else:
            raise ValueError(f"未知损失函数: {loss_type}")

    def _get_optimizer(self):
        """获取优化器"""
        opt_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if opt_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"未知优化器: {opt_type}")

    def _get_scheduler(self):
        """获取学习率调度器"""
        sched_type = self.config.get('scheduler', 'cosine')

        if sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=0.1
            )
        elif sched_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10
            )
        else:
            return None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 记录到tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_dices = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            # 计算Dice
            dice = TorchSegmentationMetrics.dice_score(outputs, labels)
            all_dices.append(dice.mean().item())

        avg_loss = total_loss / len(self.val_loader)
        avg_dice = np.mean(all_dices)

        return avg_loss, avg_dice

    def train(self):
        """完整训练流程"""
        print(f"[INFO] 开始训练，设备: {self.device}")
        print(f"[INFO] 训练样本: {len(self.train_loader.dataset)}")
        print(f"[INFO] 验证样本: {len(self.val_loader.dataset)}")

        num_epochs = self.config.get('epochs', 100)
        early_stop_patience = self.config.get('early_stop_patience', 20)
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss, val_dice = self.validate()

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

            # 记录
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('val/dice', val_dice, epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

            # 保存最佳模型
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint('best_model.pth')
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停
            if no_improve_epochs >= early_stop_patience:
                print(f"[INFO] 早停: {early_stop_patience} 个epoch无改善")
                break

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        print(f"[INFO] 训练完成，最佳Dice: {self.best_dice:.4f}")
        self.writer.close()

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config
        }

        torch.save(checkpoint, self.save_dir / filename)
        print(f"[INFO] 已保存检查点: {filename}")

    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_dice = checkpoint['best_dice']
        print(f"[INFO] 已加载检查点: {filename}")


# 使用示例配置
TRAINING_CONFIG = {
    # 数据
    'batch_size': 4,
    'num_workers': 4,
    'patch_size': (128, 128, 128),

    # 模型
    'in_channels': 1,
    'num_classes': 4,
    'model_type': 'unet3d',

    # 训练
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'loss': 'combined',

    # 其他
    'use_amp': True,
    'early_stop_patience': 20,
    'save_dir': 'checkpoints',
    'log_dir': 'runs',
}
```

---

## 附录：推荐工具库

| 用途 | 推荐库 | 说明 |
|-----|-------|------|
| 医学图像IO | `SimpleITK`, `nibabel`, `pydicom` | DICOM/NIfTI读写 |
| 3D数据处理 | `MONAI`, `TorchIO` | 3D医学图像专用 |
| 数据增强 | `albumentations`, `MONAI` | 2D/3D增强 |
| 模型实现 | `MONAI`, `segmentation-models-pytorch` | 预训练模型 |
| 可视化 | `matplotlib`, `itkwidgets`, `napari` | 医学图像可视化 |
| 可解释性 | `captum` | 模型解释 |
| 实验管理 | `wandb`, `tensorboard`, `mlflow` | 训练跟踪 |

---

## 总结

本规范清单涵盖了CV和医学图像深度学习训练的核心方面：

1. **训练注意事项**：确定性设置、类别不平衡处理
2. **数据预处理**：窗宽窗位、重采样、归一化
3. **数据增强**：几何变换、强度变换、弹性形变
4. **评估指标**：Dice、IoU、HD95、AUC等
5. **内存优化**：补丁采样、混合精度、梯度累积
6. **多模态处理**：模态融合、缺失处理
7. **伦理合规**：隐私保护、审计追踪、可解释性

建议根据具体任务需求选择适用的策略和代码模块。
