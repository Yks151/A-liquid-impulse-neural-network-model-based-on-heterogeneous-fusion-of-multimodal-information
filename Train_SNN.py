import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.cuda.amp import autocast, GradScaler
import os
import warnings
from torchvision.transforms import GaussianBlur, RandomApply
from utils import read_directory
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
warnings.filterwarnings("ignore")

# -------------------- 改进的SNN核心模块 --------------------
class SurrogateGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 0.2 * F.relu(1 - torch.abs(x/2))  # 改进的梯度近似
        return grad * grad_output

class LIFNeuron(nn.Module):
    """改进的LIF神经元，使用替代梯度"""
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.spike_grad = SurrogateGradFn.apply
        self.reset()
        
    def reset(self):
        self.mem_potential = None
        
    def forward(self, x):
        if self.mem_potential is None or self.mem_potential.size(0) != x.size(0):
            self.mem_potential = torch.zeros_like(x)
        
        new_potential = self.decay * self.mem_potential + x
        spike = self.spike_grad(new_potential - self.threshold)
        self.mem_potential = (new_potential - spike * self.threshold).detach()
        return spike

class SNNConvBlock(nn.Module):
    """时空特征提取块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron()
        self.attn = TemporalAttention(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.attn(x)
        return self.lif(x)

class TemporalAttention(nn.Module):
    """时空注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ca = self.channel_att(x)
        return x * ca

# -------------------- 优化的模型架构 --------------------
class EfficientSNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            SNNConvBlock(3, 32),  # 减少初始通道数
            nn.AvgPool2d(2),
            SNNConvBlock(32, 64),
            nn.AvgPool2d(2),
            SNNConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 512),  # 减少全连接层维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):  # 保留这一个forward方法
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -------------------- 数据增强与预处理 --------------------
class TimeFreqAugment(nn.Module):
    """时频域数据增强"""
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            RandomApply([GaussianBlur(5), ChannelShuffle()], p=0.6),
            RandomApply([nn.Dropout(0.2)], p=0.3)
        )
        
    def forward(self, x):
        return self.transform(x)

class ChannelShuffle(nn.Module):
    def forward(self, x):
        return x[:, torch.randperm(x.size(1))]

def preprocess(data):
    data = torch.tensor(data).float()
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    return (data - mean) / std

# -------------------- 训练工具 --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()

# -------------------- 可视化函数 --------------------
# 修改generate_report函数中的样式设置
def generate_report(model, loaders, class_names, filename="diagnostic_report.pdf"):
    """生成综合诊断报告"""
    plt.style.use('seaborn-v0_8')  # 使用兼容的样式名称
    sns.set_palette("husl")
    
    with PdfPages(filename) as pdf:
        # 训练曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(train_loss, label='Train', lw=2)
        ax1.plot(valid_loss, label='Validation', lw=2)
        ax1.set_title('Training Curve', fontsize=14)
        ax1.legend()
        
        ax2.plot(train_acc, label='Train')
        ax2.plot(valid_acc, label='Validation')
        ax2.set_title('Accuracy Curve', fontsize=14)
        pdf.savefig(fig)
        plt.close()
        
        # 特征可视化
        fig = plt.figure(figsize=(16, 6))
        features, labels = extract_features(model, loaders['valid'])
        tsne = TSNE(n_components=2).fit_transform(features[:1000])
        
        ax = fig.add_subplot(121)
        sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=labels[:1000], palette="husl", ax=ax)
        ax.set_title('t-SNE Feature Projection')
        
        # 混淆矩阵
        ax = fig.add_subplot(122)
        probs, preds, true = get_predictions(model, loaders['test'])
        cm = confusion_matrix(true, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        pdf.savefig(fig)
        plt.close()
        
        # 脉冲活动可视化
        fig = plt.figure(figsize=(12, 6))
        # 在generate_report函数前添加plot_spike_activity函数定义
        def plot_spike_activity(model, loader):
            """可视化脉冲活动"""
            model.eval()
            samples, _ = next(iter(loader))
            samples = samples[:4].to(device)  # 取前4个样本
            
            # 获取各层脉冲活动
            activations = []
            def hook(module, input, output):
                activations.append(output.detach().cpu().numpy())
            
            handles = []
            for layer in model.feature_extractor:
                if hasattr(layer, 'lif'):
                    handles.append(layer.lif.register_forward_hook(hook))
            
            with torch.no_grad():
                _ = model(samples)
            
            # 移除hook
            for handle in handles:
                handle.remove()
            
            # 绘制脉冲活动
            plt.figure(figsize=(12, 6))
            for i, act in enumerate(activations):
                plt.subplot(1, len(activations), i+1)
                plt.imshow(act[0,0], cmap='hot')
                plt.title(f'Layer {i+1}')
                plt.axis('off')
            plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

def extract_features(model, loader):
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feat = model.feature_extractor(x.to(device)).flatten(1).cpu()
            features.append(feat)
            labels.append(y)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()

def get_predictions(model, loader):
    probs, preds, true = [], [], []
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            probs.append(F.softmax(output, 1).cpu())
            preds.append(output.argmax(1).cpu())
            true.append(y)
    return torch.cat(probs).numpy(), torch.cat(preds).numpy(), torch.cat(true).numpy()

# -------------------- 主训练流程 --------------------
if __name__ == '__main__':
    x_test, y_test = read_directory('小波时频/test_img', 64, 64, True)

    # 将标签转换为整数类型
    y_train = y_train.astype(np.int64)
    y_valid = y_valid.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 添加类别分布检查
    print("训练集类别分布:", np.bincount(y_train))
    print("验证集类别分布:", np.bincount(y_valid))

    # 数据预处理
    train_data = preprocess(x_train)
    valid_data = preprocess(x_valid)
    test_data = preprocess(x_test)

    # 创建数据加载器
    loaders = {
        'train': DataLoader(TensorDataset(TimeFreqAugment()(train_data), torch.tensor(y_train).long()), batch_size=128, shuffle=True),
        'valid': DataLoader(TensorDataset(valid_data, torch.tensor(y_valid).long()), batch_size=128),
        'test': DataLoader(TensorDataset(test_data, torch.tensor(y_test).long()), batch_size=128)
    }

    # 初始化模型
    model = EfficientSNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 增大学习率
    criterion = FocalLoss(gamma=2, alpha=0.25)  # 调整Focal Loss参数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 修改学习率调度
    criterion = FocalLoss()
    scaler = GradScaler()

    # 训练循环
    best_acc = 0
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    def evaluate(model, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                correct += (output.argmax(1) == y).sum().item()
        return 100 * correct / len(loader.dataset)
    
    # 修改训练循环部分
    for epoch in range(100):
        model.train()
        total_loss = correct = 0
        
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
        
        # 计算训练指标
        train_loss.append(total_loss/len(loaders['train']))
        train_acc.append(100*correct/len(loaders['train'].dataset))
        
        # 验证阶段
        val_acc = evaluate(model, loaders['valid'])
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss[-1]:.4f} | Acc: {train_acc[-1]:.2f}% | Val Acc: {val_acc:.2f}%")

    # 生成报告
    model.load_state_dict(torch.load('best_model.pth'))
    generate_report(model, loaders, class_names=["class_0",...,"class_9"])

    # 测试评估
    probs, preds, true = get_predictions(model, loaders['test'])
    print(classification_report(true, preds))
    print("Test Accuracy:", 100*(preds == true).mean())