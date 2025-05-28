import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import read_directory
from torch.cuda.amp import GradScaler
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
# -------------------- 核心模块 --------------------
class SurrogateGradFn(torch.autograd.Function):
    """替代梯度函数"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 0.2 * F.relu(1 - torch.abs(x/2))
        return grad * grad_output

class DynamicODE(nn.Module):
    """LNN动力学系统"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.time_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, t, x):
        time_factor = 1.0 / (1.0 + self.time_gate(x))
        dx = F.gelu(x @ self.weight) * time_factor
        return dx

# 在文件开头导入部分添加
from torch.amp import autocast, GradScaler
# 在文件开头添加导入
from torch import Tensor
from typing import Optional

# 添加DropPath类实现
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
from torchvision import transforms

# 添加TimeFreqAugment类
class TimeFreqAugment:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10)
        ])

    def __call__(self, x):
        if len(x.shape) == 3:  # (C,H,W)
            return self.transform(x)
        elif len(x.shape) == 4:  # (B,C,H,W)
            return torch.stack([self.transform(img) for img in x])
        return x

# 修改LIFNeuron类
class LIFNeuron(nn.Module):
    def __init__(self, threshold=0.3, decay=0.85, ode_dim=64):  # 降低阈值，调整衰减
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.ode = DynamicODE(ode_dim)
        self.spike_grad = SurrogateGradFn.apply
        self.reset()  # 调用reset方法初始化

    def reset(self):
        """重置膜电位"""
        self.mem_potential = None

    def forward(self, x):
        if self.mem_potential is None:
            self.mem_potential = torch.zeros_like(x)

        t = torch.linspace(0, 1, 3).to(x.device)  # 增加时间点数量
        ode_out = odeint(self.ode, x, t,
                        method='dopri5',
                        rtol=1e-2,  # 增大容差减少计算量
                        atol=1e-3,
                        options={'min_step': 1e-2, 'max_step': 0.1})[-1]  # 调整步长范围

        new_potential = self.decay * self.mem_potential + ode_out
        spike = self.spike_grad(new_potential - self.threshold)
        self.mem_potential = (new_potential - spike * self.threshold).detach()
        return spike

# 添加注意力模块
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.channel_att(x)
        return x * att

# 修改LNN_SNN_Block
class LNN_SNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # 添加谱归一化
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        # 添加更强的DropPath正则化
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = LIFNeuron(ode_dim=out_channels)
        self.att = AttentionBlock(out_channels)  # 添加注意力
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = LIFNeuron(ode_dim=out_channels, threshold=0.5, decay=0.8)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.lif1(self.bn1(self.conv1(x)))
        x = self.lif2(self.bn2(self.conv2(x)))
        return x + residual  # 残差连接

class LNN_SNN(nn.Module):
    def __init__(self, input_shape=(3,64,64), num_classes=10):
        super().__init__()
        # 添加初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 修改分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 1024),  # 减少参数量
            nn.BatchNorm1d(1024),
            nn.SiLU(),  # 使用更平滑的激活函数
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        c, h, w = input_shape

        self.feature_extractor = nn.Sequential(
            LNN_SNN_Block(3, 64),
            nn.AvgPool2d(2),
            LNN_SNN_Block(64, 128),
            nn.AvgPool2d(2),
            LNN_SNN_Block(128, 256),
            nn.AvgPool2d(2),
            LNN_SNN_Block(256, 512),
            nn.AdaptiveAvgPool2d((2,2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(1)
        return self.classifier(x)