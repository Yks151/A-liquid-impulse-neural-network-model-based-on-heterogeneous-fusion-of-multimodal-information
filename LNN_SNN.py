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
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# 修改LIFNeuron类
class LIFNeuron(nn.Module):
    def __init__(self, threshold=0.3, decay=0.85, ode_dim=64):  # 降低阈值，调整衰减
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.ode = DynamicODE(ode_dim)
        self.spike_grad = SurrogateGradFn.apply
        self.reset()
        
    def forward(self, x):
        if self.mem_potential is None:
            self.mem_potential = torch.zeros_like(x)
            
        t = torch.linspace(0, 1, 3).to(x.device)  # 增加时间点数量
        ode_out = odeint(self.ode, x, t,
                        method='dopri5',
                        rtol=1e-3,  # 放宽容差
                        atol=1e-4,
                        options={'min_step': 1e-3, 'max_step': 0.2})[-1]
        
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
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
    """改进的完整模型"""
    def __init__(self, input_shape=(3,64,64), num_classes=10):
        super().__init__()
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

# 准备数据
def preprocess(data):
    """数据预处理函数"""
    # 将4D数据转换为2D (样本数, 通道*高度*宽度)
    original_shape = data.shape
    data = data.reshape(original_shape[0], -1)  # 展平除第一维外的所有维度
    
    # 标准化处理
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # 恢复原始形状
    data = data.reshape(original_shape)
    
    # 转换为适合SNN的脉冲序列
    spike_data = np.where(data > 0.5, 1, 0)
    
    # 转换为PyTorch张量
    return torch.FloatTensor(spike_data)

def prepare_data():
    """准备训练、验证和测试数据"""
    x_train, y_train = read_directory('小波时频/train_img', 64, 64, True)
    x_valid, y_valid = read_directory('小波时频/valid_img', 64, 64, True)
    x_test, y_test = read_directory('小波时频/test_img', 64, 64, True)
    
    train_data = preprocess(x_train)
    valid_data = preprocess(x_valid)
    test_data = preprocess(x_test)
    
    loaders = {
        'train': DataLoader(TensorDataset(TimeFreqAugment()(train_data), torch.tensor(y_train).long()), 
                  batch_size=64, shuffle=True),
        'valid': DataLoader(TensorDataset(valid_data, torch.tensor(y_valid).long()),
                  batch_size=64),
        'test': DataLoader(TensorDataset(test_data, torch.tensor(y_test).long()),
                 batch_size=64)
    }
    return loaders

# 修改训练部分
def train_model(model, loaders, optimizer, criterion, scheduler, scaler, epochs=100):
    # 学习率预热
    warmup_epochs = 5
    for epoch in range(epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            lr = 0.0001 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        model.train()
        total_loss = correct = 0
        
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(x)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            correct += (output.argmax(1) == y).sum().item()
        
        # 计算训练指标
        train_loss.append(total_loss/len(loaders['train']))
        train_acc.append(100*correct/len(loaders['train'].dataset))
        
        # 验证阶段
        val_acc = evaluate(model, loaders['valid'])
        valid_loss.append(0)  # 可根据需要添加验证损失计算
        valid_acc.append(val_acc)
        
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_lnn_snn.pth')
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss[-1]:.4f} | Acc: {train_acc[-1]:.2f}% | Val Acc: {val_acc:.2f}%")
    
    return train_loss, train_acc, valid_loss, valid_acc

def evaluate(model, loader):
    """评估模型性能"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            output = model(x.to(device))
            correct += (output.argmax(1) == y.to(device)).sum().item()
    return 100 * correct / len(loader.dataset)

def main():
    """主函数"""
    # 准备数据
    loaders = prepare_data()
    
    # 初始化模型
    model = LNN_SNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    scaler = GradScaler()
    
    # 训练模型
    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, loaders, optimizer, criterion, scheduler, scaler)
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load('best_lnn_snn.pth'))
    test_acc = evaluate(model, loaders['test'])
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 生成可视化结果
    generate_visualizations(model, loaders, train_loss, train_acc, valid_loss, valid_acc)

if __name__ == '__main__':
    main()

    # 修改generate_visualizations函数
    # 修改generate_visualizations函数中的样式设置
    def generate_visualizations(model, loaders, train_loss, train_acc, valid_loss, valid_acc, filename="results.pdf"):
        """生成可视化图表并保存结果"""
        plt.style.use('seaborn-v0_8')  # 更新为新的样式名称
        
        # 创建PDF文件保存所有图表
        with PdfPages(filename) as pdf:
            # 1. 训练和验证损失曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_loss, label='Train Loss', color='blue')
            ax.plot(valid_loss, label='Valid Loss', color='orange')
            
            # 添加置信区间
            train_loss_smooth = pd.Series(train_loss).rolling(5, min_periods=1).mean()
            valid_loss_smooth = pd.Series(valid_loss).rolling(5, min_periods=1).mean()
            ax.fill_between(range(len(train_loss)), 
                           train_loss_smooth*0.95, 
                           train_loss_smooth*1.05,
                           color='blue', alpha=0.2)
            ax.fill_between(range(len(valid_loss)),
                           valid_loss_smooth*0.95,
                           valid_loss_smooth*1.05,
                           color='orange', alpha=0.2)
            
            ax.set_title('Training and Validation Loss')
            ax.legend()
            pdf.savefig(fig)
            plt.close()
            
            # 2. 训练和验证准确率曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_acc, label='Train Accuracy', color='green')
            ax.plot(valid_acc, label='Valid Accuracy', color='red')
            
            # 添加置信区间
            train_acc_smooth = pd.Series(train_acc).rolling(5, min_periods=1).mean()
            valid_acc_smooth = pd.Series(valid_acc).rolling(5, min_periods=1).mean()
            ax.fill_between(range(len(train_acc)),
                           train_acc_smooth*0.98,
                           train_acc_smooth*1.02,
                           color='green', alpha=0.2)
            ax.fill_between(range(len(valid_acc)),
                           valid_acc_smooth*0.98,
                           valid_acc_smooth*1.02,
                           color='red', alpha=0.2)
            
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            pdf.savefig(fig)
            plt.close()
            
            # 保存结果到Excel
            results = pd.DataFrame({
                'Epoch': range(1, len(train_loss)+1),
                'Train Loss': train_loss,
                'Valid Loss': valid_loss,
                'Train Acc': train_acc,
                'Valid Acc': valid_acc
            })
            results.to_excel('training_results.xlsx', index=False)
            
            # 保存测试结果
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for x, y in loaders['test']:
                    output = model(x.to(device))
                    test_correct += (output.argmax(1) == y.to(device)).sum().item()
            
            test_acc = 100 * test_correct / len(loaders['test'].dataset)
            test_results = pd.DataFrame({
                'Test Accuracy': [test_acc]
            })
            test_results.to_excel('test_results.xlsx', index=False)
            
            # 保存模型
            torch.save(model.state_dict(), 'LNN_SNN_model.pth')
