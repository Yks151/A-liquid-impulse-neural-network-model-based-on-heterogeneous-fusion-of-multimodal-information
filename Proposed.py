import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler  # 保持原有导入方式
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# coding: utf-8
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from utils import read_directory
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
from scipy.io import loadmat,savemat
torch.manual_seed(0)
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # 设置为实际可用的CPU核心数

# In[2] 加载数据
num_classes=10
height=64
width=64
# In[3]: 参数设置
learning_rate = 0.005#学习率
num_epochs = 200#迭代次数
batch_size = 64 #batchsize

# -------------------- 核心模块 --------------------
class SurrogateGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 0.2 * F.relu(1 - torch.abs(x/2))
        return grad * grad_output

class LIFNeuron(nn.Module):
    """改进的LIF神经元"""
    def __init__(self, threshold=1.0, decay=0.95):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.spike_grad = SurrogateGradFn.apply
        self.reset()
        
    def reset(self):
        self.mem_potential = None
        
    def forward(self, x):
        if self.mem_potential is None or self.mem_potential.shape != x.shape:
            self.mem_potential = torch.zeros_like(x)
            
        new_potential = self.decay * self.mem_potential + (1-self.decay)*x
        spike = self.spike_grad(new_potential - self.threshold)
        self.mem_potential = (new_potential - spike * self.threshold).detach()
        return spike

class LiquidLayer(nn.Module):
    """动态特征编码器"""
    def __init__(self, in_channels=3, hidden_dim=32, tau=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.Wx = nn.Parameter(torch.randn(in_channels, hidden_dim) * 0.1)
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        
        chunk_size = 16
        h = torch.zeros(B, H*W, self.hidden_dim, device=x.device)
        
        for t in range(0, H*W, chunk_size):
            chunk_end = min(t + chunk_size, H*W)
            input_chunk = torch.matmul(x_flat[:, t:chunk_end, :], self.Wx)
            recurrent_chunk = torch.matmul(h[:, t:chunk_end, :], self.Wh)
            current_h = h[:, t:chunk_end, :]
            dh = (-current_h + F.relu(input_chunk + recurrent_chunk + self.bias)) / self.tau
            h[:, t:chunk_end, :] = current_h + dh * self.alpha

        return h.permute(0, 2, 1).view(B, -1, H, W)

class SNNConvBlock(nn.Module):
    """时空特征提取块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.lif(x)

# -------------------- 混合模型架构 --------------------
class HybridDiagnosisModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 图像处理分支 (采用简化版本)
        self.image_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        
        # 信号处理分支 (采用简化版本)
        self.signal_stream = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 融合分类器 (保留dropout和更合理的维度)
        self.classifier = nn.Sequential(
            nn.Linear(64*4*4 + 64*16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        # 并行处理两个模态
        x1 = self.image_stream(x1).flatten(1)
        x2 = self.signal_stream(x2).flatten(1)
        
        # 特征融合
        x = torch.cat([x1, x2], dim=1)
        
        # 分类
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        
        # 返回所有需要的输出
        return logits, probas, x1, x2, x

# -------------------- 训练配置 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 文件开头添加导入
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 修改训练配置部分
learning_rate = 0.001  # 降低学习率
num_epochs = 100  # 减少epoch数量
batch_size = 32  # 减小batch size
# 修改模型初始化部分
model = HybridDiagnosisModel(num_classes=10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # 改用基于验证准确率的调度器

# 在训练配置部分添加变量初始化
train_loss = []
train_acc = []
valid_acc = []
# 修改导入部分
from torch.amp import GradScaler, autocast

# 在数据加载部分添加测试集数据
if __name__ == '__main__':
    # 加载时频图数据
    x_train, y_train = read_directory('小波时频/train_img', 64, 64, True)
    x_valid, y_valid = read_directory('小波时频/valid_img', 64, 64, True)
    x_test, y_test = read_directory('小波时频/test_img', 64, 64, True)  # 添加测试集加载    
    # 类型转换
    y_train = y_train.astype(np.int64)
    y_valid = y_valid.astype(np.int64)
    y_test = y_test.astype(np.int64)  # 添加测试集标签转换    
    # 加载FFT数据
    datafft = loadmat('FFT频谱/FFT.mat')
    x_train2 = datafft['train_X']
    x_valid2 = datafft['valid_X']
    x_test2 = datafft['test_X']
    # 在数据标准化部分修改
    ss2 = StandardScaler().fit(x_train2)
    x_train2 = np.nan_to_num(ss2.transform(x_train2), nan=0.0, posinf=0.0, neginf=0.0)
    x_valid2 = np.nan_to_num(ss2.transform(x_valid2), nan=0.0, posinf=0.0, neginf=0.0)
    x_test2 = np.nan_to_num(ss2.transform(x_test2), nan=0.0, posinf=0.0, neginf=0.0)  
    
    x_train = x_train / 255.0
    x_valid = x_valid / 255.0 
    x_test = x_test / 255.0
    # 调整维度
    x_train2 = x_train2.reshape(x_train2.shape[0], 1, -1)
    x_valid2 = x_valid2.reshape(x_valid2.shape[0], 1, -1)
    x_test2 = x_test2.reshape(x_test2.shape[0], 1, -1)
    # 转换为Tensor
    train_features = torch.tensor(x_train).float()
    valid_features = torch.tensor(x_valid).float()
    test_features = torch.tensor(x_test).float()
    train_features2 = torch.tensor(x_train2).float()
    valid_features2 = torch.tensor(x_valid2).float()
    test_features2 = torch.tensor(x_test2).float()
    train_labels = torch.tensor(y_train).long()
    valid_labels = torch.tensor(y_valid).long()
    test_labels = torch.tensor(y_test).long()

    # -------------------- 数据加载器 --------------------
    batch_size = 128
    loaders = {
        'train': DataLoader(TensorDataset(train_features, train_features2, train_labels),
                          batch_size=batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(TensorDataset(valid_features, valid_features2, valid_labels),
                          batch_size=batch_size, pin_memory=True),
        'test': DataLoader(TensorDataset(test_features, test_features2, test_labels),
                         batch_size=batch_size)
    }

    # -------------------- 模型初始化 --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 修改模型初始化
    model = HybridDiagnosisModel(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    
    # 修改损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    scaler = GradScaler()

    # -------------------- 训练循环 --------------------
    num_epochs = 200
    # 在训练循环中添加
    best_valid_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0  # 在这里定义total变量
    
        for x1, x2, y in loaders['train']:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits, _, _, _, _ = model(x1, x2)
                loss = criterion(logits, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        # 计算并记录训练准确率
        current_train_acc = 100 * correct / total if total > 0 else 0.0
        train_acc.append(current_train_acc)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x1, x2, y in loaders['valid']:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits, _, _, _, _ = model(x1, x2)
                _, predicted = logits.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        # 记录指标
        train_loss.append(epoch_loss/len(loaders['train']))
        valid_acc.append(100*val_correct/val_total if val_total > 0 else 0.0)
        
        # 学习率调整
        scheduler.step(valid_acc[-1])  # 传入验证准确率作为metrics

        # 保存最佳模型
        if valid_acc[-1] > best_valid_acc:
            best_valid_acc = valid_acc[-1]
            patience_counter = 0  # 重置计数器
            torch.save(model.state_dict(), 'best_lnn_snn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            print(f"Epoch {epoch+1}: New best model saved! Val Acc: {best_valid_acc:.2f}%")

        # 打印进度
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss[-1]:.4f} | "
              f"Train Acc: {train_acc[-1]:.2f}% | "
              f"Val Acc: {valid_acc[-1]:.2f}%")

    # -------------------- 测试评估 --------------------
    model.load_state_dict(torch.load('best_lnn_snn_model.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x1, x2, y in loaders['test']:
            x1, x2 = x1.to(device), x2.to(device)
            logits, _, _, _, _ = model(x1, x2)
            _, predicted = logits.max(1)

            test_total += y.size(0)
            test_correct += predicted.cpu().eq(y).sum().item()
            y_true.extend(y.numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(f"\nTest Accuracy: {100*test_correct/test_total:.2f}%")
    print(classification_report(y_true, y_pred, target_names=[f"Class_{i}" for i in range(10)]))

    # -------------------- 可视化报告 --------------------
    # 特征可视化
    def extract_features(model, loader):
        features_img, features_sig, labels = [], [], []
        model.eval()
        with torch.no_grad():
            for x1, x2, y in loader:
                x1, x2 = x1.to(device), x2.to(device)
                _, _, img_feat, sig_feat, _ = model(x1, x2)
                features_img.append(img_feat.cpu())
                features_sig.append(sig_feat.cpu())
                labels.append(y)
        return torch.cat(features_img).numpy(), torch.cat(features_sig).numpy(), torch.cat(labels).numpy()

    # 先提取特征
    features_img, features_sig, labels = extract_features(model, loaders['test'])
    
    # 添加数据检查和处理
    features_img = np.nan_to_num(features_img, nan=0.0, posinf=0.0, neginf=0.0)
    features_sig = np.nan_to_num(features_sig, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 然后再进行t-SNE可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    tsne_img = TSNE(n_components=2, init='pca', random_state=0, perplexity=min(30, len(features_img)-1)).fit_transform(features_img)
    tsne_img = np.nan_to_num(tsne_img, nan=0.0, posinf=0.0, neginf=0.0)
    plt.scatter(tsne_img[:,0], tsne_img[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.title('Time-Frequency Features')

    plt.subplot(122)
    tsne_sig = TSNE(n_components=2, init='pca', random_state=0, perplexity=min(30, len(features_sig)-1)).fit_transform(features_sig)
    tsne_sig = np.nan_to_num(tsne_sig, nan=0.0, posinf=0.0, neginf=0.0)
    plt.scatter(tsne_sig[:,0], tsne_sig[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.title('FFT Signal Features')
    plt.tight_layout()
    plt.show()

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f"Class_{i}" for i in range(10)],
               yticklabels=[f"Class_{i}" for i in range(10)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(train_loss, 'b-', lw=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(122)
    plt.plot(train_acc, 'r--', lw=2, label='Train')
    plt.plot(valid_acc, 'g-.', lw=2, label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存完整报告
    generate_diagnostic_report(model, loaders, [f"Class_{i}" for i in range(10)])
# 在数据加载后添加
print("训练集类别分布:", np.bincount(y_train))
print("验证集类别分布:", np.bincount(y_valid))
print("测试集类别分布:", np.bincount(y_test))

# -------------------- 核心模块重设计 --------------------
class ImprovedLIFNeuron(nn.Module):
    """改进的LIF神经元"""
    def __init__(self, threshold=0.5, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.spike_grad = SurrogateGradFn.apply
        self.reset()
        
    def reset(self):
        self.mem_potential = None
        
    def forward(self, x):
        if self.mem_potential is None:
            self.mem_potential = torch.zeros_like(x[:,0])
            
        # 多时间步处理
        spikes = []
        for t in range(x.shape[1]):
            new_potential = self.decay * self.mem_potential + x[:,t]
            spike = self.spike_grad(new_potential - self.threshold)
            self.mem_potential = (new_potential - spike * self.threshold).detach()
            spikes.append(spike)
            
        return torch.stack(spikes, dim=1)

# -------------------- 模型架构重设计 --------------------
class HybridDiagnosisModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 图像处理分支
        self.image_stream = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        
        # 信号处理分支
        self.signal_stream = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(64*4*4 + 64*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        x1 = self.image_stream(x1)
        x1 = x1.flatten(1)
        
        x2 = self.signal_stream(x2)
        x2 = x2.flatten(1)
        
        x = torch.cat([x1, x2], dim=1)
        return self.classifier(x)

# -------------------- 训练配置优化 --------------------
# 修改数据预处理部分
def preprocess_data():
    # 图像数据
    x_train, y_train = read_directory('小波时频/train_img', 64, 64, True)
    x_valid, y_valid = read_directory('小波时频/valid_img', 64, 64, True)
    x_test, y_test = read_directory('小波时频/test_img', 64, 64, True)
    
    # 确保数据范围在[0,1]之间
    x_train = np.clip(x_train / 255.0, 0, 1)
    x_valid = np.clip(x_valid / 255.0, 0, 1)
    x_test = np.clip(x_test / 255.0, 0, 1)
    
    # FFT数据处理
    datafft = loadmat('FFT频谱/FFT.mat')
    x_train2 = np.nan_to_num(datafft['train_X'], nan=0.0)
    x_valid2 = np.nan_to_num(datafft['valid_X'], nan=0.0)
    x_test2 = np.nan_to_num(datafft['test_X'], nan=0.0)
    
    # 标准化FFT数据
    scaler = StandardScaler()
    x_train2 = scaler.fit_transform(x_train2)
    x_valid2 = scaler.transform(x_valid2)
    x_test2 = scaler.transform(x_test2)
    
    # 转换为Tensor
    train_set = TensorDataset(
        torch.FloatTensor(x_train),
        torch.FloatTensor(x_train2).unsqueeze(1),
        torch.LongTensor(y_train)
    )
    valid_set = TensorDataset(
        torch.tensor(x_valid).float(),
        torch.tensor(x_valid2).float().unsqueeze(1),
        torch.tensor(y_valid).long()
    )
    test_set = TensorDataset(
        torch.tensor(x_test).float(),
        torch.tensor(x_test2).float().unsqueeze(1),
        torch.tensor(y_test).long()
    )
    
    return {
        'train': DataLoader(train_set, batch_size=64, shuffle=True),
        'valid': DataLoader(valid_set, batch_size=64),
        'test': DataLoader(test_set, batch_size=64)
    }

# -------------------- 训练循环优化 --------------------
def train_model():
    # 初始化
    model = HybridDiagnosisModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 训练循环
    for epoch in range(200):
        model.train()
        for x1, x2, y in loaders['train']:
            optimizer.zero_grad()
            output = model(x1.to(device), x2.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optimizer.step()
        
        # 验证和模型保存...
        scheduler.step()
class SimplifiedHybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 图像分支
        self.image_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        
        # 信号分支
        self.signal_net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64*4*4 + 64*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x1, x2):
        x1 = self.image_net(x1).flatten(1)
        x2 = self.signal_net(x2).flatten(1)
        x = torch.cat([x1, x2], dim=1)
        return self.classifier(x)

# 训练配置修改
model = SimplifiedHybridModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 训练循环中添加梯度检查
for epoch in range(num_epochs):
    model.train()
    for x1, x2, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x1.to(device), x2.to(device))
        loss = criterion(outputs, y.to(device))
        loss.backward()
        
        # 打印梯度信息
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1./2)
        print(f"Gradient norm: {total_norm:.4f}")
        
        optimizer.step()