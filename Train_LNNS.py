import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import read_directory
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 用于调试CUDA错误

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# -------------------- 自定义模块 --------------------
class StochasticDepth(nn.Module):
    """随机深度正则化"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        survival_rate = 1 - self.p
        mask = torch.rand(x.shape[0], 1, device=x.device) < survival_rate
        return x * mask / survival_rate

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):  # 改为监控准确率
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(log_probs * (1 - self.eps + self.eps / num_classes)).sum(dim=-1)
        return loss.mean()

# -------------------- 液态神经网络模型 --------------------
class MultiScaleLiquid(nn.Module):
    """多尺度特征融合层"""
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 多尺度分支
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim//4),
                nn.GELU(),
                nn.LayerNorm(hidden_dim//4)
            ) for _ in range(4)
        ])

        # 动态特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU()
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.drop_path = StochasticDepth(p=0.2)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Identity()

    def forward(self, x):
        # 多尺度处理
        features = [branch(x) for branch in self.scale_branches]
        fused = self.fusion(torch.cat(features, dim=1))

        # 注意力增强
        attn_out, _ = self.attention(fused.unsqueeze(1), fused.unsqueeze(1), fused.unsqueeze(1))
        attn_out = self.drop_path(attn_out.squeeze(1))
        
        return self.norm(x + attn_out)

# 修改DynamicODE类
class DynamicODE(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        # 使用更稳定的权重初始化
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        
    def forward(self, t, x):
        time_constant = 1.0 / (1.0 + torch.sigmoid(self.gate(x)))
        dx = F.gelu(x @ self.weight) * time_constant
        return dx

# 修改EnhancedLNN类中的ODE系统初始化
class EnhancedLNN(nn.Module):
    def __init__(self, input_shape=(3,64,64), num_classes=10):
        super().__init__()
        c, h, w = input_shape
        self.input_dim = c * h * w
        
        # 增强的特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.4),
            MultiScaleLiquid(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2)
        )

        # 改进的ODE系统
        self.ode_system = DynamicODE(512)  # 确保这里使用512与特征提取网络输出维度一致

        # 增强分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.flatten(1)
        features = self.feature_net(x)
        
        # 优化ODE求解参数
        t = torch.linspace(0, 3, 10).to(device)  # 调整时间范围和步数
        ode_feat = odeint(self.ode_system, features, t, 
                         method='dopri5', 
                         rtol=1e-4, 
                         atol=1e-5,
                         options={'max_num_steps': 1000})[-1]
        
        logits = self.classifier(ode_feat)
        return logits, F.softmax(logits, dim=1)

# 修改权重初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('linear'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# 先创建模型实例
num_classes = 10
num_epochs = 200
height, width = 64, 64
model = EnhancedLNN(num_classes=num_classes).to(device)
model.apply(init_weights)

# 然后定义优化器
optimizer = torch.optim.AdamW(model.parameters(),
                            lr=2e-4,  # 降低学习率
                            weight_decay=0.01,
                            betas=(0.9, 0.999))

# 修改学习率调度器配置
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # 改为监控准确率
    factor=0.5,
    patience=3,  # 减少耐心值
    min_lr=1e-6  # 降低最小学习率
    # 移除verbose参数
)

# -------------------- 数据加载与预处理 --------------------

# 加载数据
x_train, y_train = read_directory('小波时频/train_img', height, width, normal=1)
x_valid, y_valid = read_directory('小波时频/valid_img', height, width, normal=1)
x_test, y_test = read_directory('小波时频/test_img', height, width, normal=1)

# 转换为Tensor
train_features = torch.tensor(x_train).float()
train_labels = torch.tensor(y_train).long()
valid_features = torch.tensor(x_valid).float()
valid_labels = torch.tensor(y_valid).long()
test_features = torch.tensor(x_test).float()
test_labels = torch.tensor(y_test).long()

# 数据标准化
mean = train_features.mean(dim=(0, 2, 3), keepdim=True)
std = train_features.std(dim=(0, 2, 3), keepdim=True)
train_features = (train_features - mean) / std
valid_features = (valid_features - mean) / std
test_features = (test_features - mean) / std

# 数据增强
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
        
    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.fn(x)
        return x

# 增强数据增强
transform = nn.Sequential(
    RandomApply(lambda x: x + torch.randn_like(x)*0.02, p=0.5),
    RandomApply(lambda x: x * (1 + torch.randn(1)*0.05), p=0.3),
    RandomApply(lambda x: torch.flip(x, [3]), p=0.3),
    RandomApply(lambda x: F.adaptive_avg_pool2d(x, (64, 64)), p=0.2)
)

# 创建DataLoader (放在数据预处理之后)
batch_size = 64
train_dataset = TensorDataset(transform(train_features), train_labels)
valid_dataset = TensorDataset(valid_features, valid_labels)
test_dataset = TensorDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

# 在文件开头添加导入
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # 兼容旧版本

# 在模型初始化后定义损失函数
criterion = nn.CrossEntropyLoss()

# 修改混合精度训练配置部分
try:
    # 新版PyTorch
    scaler = torch.amp.GradScaler() if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
except:
    # 旧版PyTorch
    scaler = torch.cuda.amp.GradScaler()

# 在训练循环中修改为:
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast(device_type='cuda' if hasattr(torch.amp, 'autocast') else None):
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # 梯度裁剪和监控
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isnan(grad_norm):
            print("梯度出现NaN值!")
            optimizer.zero_grad()
            continue
            
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    # 验证阶段
    # 在训练前添加
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # 修改训练循环中的验证部分
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            val_loss += criterion(logits, targets).item()
            _, preds = torch.max(logits, 1)
            val_correct += (preds == targets).sum().item()
            val_total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    # 在训练循环前初始化记录变量
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    best_val_acc = 0.0
    
    # 初始化早停机制
    # 修改EarlyStopping初始化部分
    early_stopper = EarlyStopping(
        patience=10,  # 适当增加耐心值
        delta=0.005  # 设置更严格的阈值
        # 移除了trace_func参数
    )
    
    train_loss.append(total_loss / len(train_loader))
    train_acc.append(correct / total * 100)
    valid_loss.append(val_loss / len(valid_loader))
    valid_acc.append(val_correct / val_total * 100)
    
    # 修改打印语句
    print(f'Epoch {epoch+1:03d}/{num_epochs} | '
    f'Train Loss: {train_loss[-1]:.4f} | Acc: {train_acc[-1]:.2f}% | '
    f'Val Loss: {valid_loss[-1]:.4f} | Val Acc: {valid_acc[-1]:.2f}% | '
    f'LR: {scheduler.get_last_lr()[0]:.2e}')  # 使用get_last_lr()

    # 记录最佳验证准确率
    if valid_acc[-1] > best_val_acc + 0.005:  # 只有当显著提升时才保存
        best_val_acc = valid_acc[-1]
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"新的最佳验证准确率: {best_val_acc:.2f}%")

    # 早停判断
    early_stopper(valid_loss[-1])
    if early_stopper.early_stop:
        print("触发早停!")
        break  # 现在这个break在正确的循环内

# -------------------- 测试评估 --------------------
# 在测试评估前添加函数定义
def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            _, preds = torch.max(logits, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total * 100

# -------------------- 测试评估 --------------------
model.load_state_dict(torch.load('best_model.pth'))
test_acc = compute_accuracy(model, test_loader)
print(f'\n测试集准确率: {test_acc:.2f}%')

# -------------------- 可视化 --------------------
plt.figure(figsize=(12,5), dpi=600)
plt.subplot(1,2,1)
plt.plot(train_loss, label='训练集', linewidth=2, color='#1f77b4')
plt.plot(valid_loss, label='验证集', linewidth=2, color='#ff7f0e')
plt.fill_between(range(len(train_loss)), 
                 np.array(train_loss)-np.std(train_loss), 
                 np.array(train_loss)+np.std(train_loss),
                 color='#1f77b4', alpha=0.2)
plt.fill_between(range(len(valid_loss)), 
                 np.array(valid_loss)-np.std(valid_loss), 
                 np.array(valid_loss)+np.std(valid_loss),
                 color='#ff7f0e', alpha=0.2)
plt.title('损失曲线', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.plot(train_acc, label='训练集', linewidth=2, color='#1f77b4')
plt.plot(valid_acc, label='验证集', linewidth=2, color='#ff7f0e')
plt.fill_between(range(len(train_acc)), 
                 np.array(train_acc)-np.std(train_acc), 
                 np.array(train_acc)+np.std(train_acc),
                 color='#1f77b4', alpha=0.2)
plt.fill_between(range(len(valid_acc)), 
                 np.array(valid_acc)-np.std(valid_acc), 
                 np.array(valid_acc)+np.std(valid_acc),
                 color='#ff7f0e', alpha=0.2)
plt.title('准确率曲线', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=600, bbox_inches='tight')
plt.show()

# 分类结果小提琴图
def plot_classification_results(model, data_loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            _, probs = model(inputs)
            all_probs.append(probs.cpu())
            all_labels.append(targets)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    plt.figure(figsize=(15, 8), dpi=600)
    for i in range(num_classes):
        class_probs = all_probs[all_labels == i, i]
        plt.violinplot(class_probs, positions=[i], showmeans=True, showmedians=True)
    
    plt.title('各类别分类置信度分布', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('预测概率', fontsize=12)
    plt.xticks(range(num_classes), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('class_prob_distribution.png', dpi=600, bbox_inches='tight')
    plt.show()

# 调用可视化函数
plot_classification_results(model, test_loader)

# 特征可视化
def visualize_features_advanced(model, data_loader, filename):
    model.eval()
    features = []
    labels = []
    probs = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            feat = model.feature_net(inputs.flatten(1))
            _, prob = model(inputs)
            features.append(feat.cpu())
            labels.append(targets)
            probs.append(prob.cpu())
    
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()

    # 3D TSNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)

    fig = plt.figure(figsize=(18, 12), dpi=300)
    
    # 3D散点图
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(reduced[:,0], reduced[:,1], reduced[:,2],
                         c=labels, cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, ax=ax1, shrink=0.6)
    ax1.set_title('3D Feature Space (t-SNE)', fontsize=12, fontweight='bold')
    
    # 2D投影
    ax2 = fig.add_subplot(222)
    scatter = ax2.scatter(reduced[:,0], reduced[:,1], c=labels, 
                         cmap='viridis', s=30, alpha=0.8)
    plt.colorbar(scatter, ax=ax2, shrink=0.6)
    ax2.set_title('2D Projection (X-Y)', fontsize=12, fontweight='bold')
    
    # 置信度分布
    ax3 = fig.add_subplot(223)
    for i in range(num_classes):
        sns.kdeplot(probs[labels==i, i], label=f'Class {i}', ax=ax3)
    ax3.set_title('Class Confidence Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Probability', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.legend(fontsize=8)
    
    # 激活强度
    ax4 = fig.add_subplot(224)
    sns.boxplot(x=labels, y=np.max(probs, axis=1), palette='Set3', ax=ax4)
    ax4.set_title('Max Activation per Class', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Class', fontsize=10)
    ax4.set_ylabel('Max Probability', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

def visualize_weights(model, filename='weight_visualization.pdf'):
    weights = []
    names = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            weights.append(param.detach().cpu().numpy())
            names.append(name)
    
    plt.figure(figsize=(15, 8), dpi=300)
    for i, (weight, name) in enumerate(zip(weights, names)):
        plt.subplot(2, 3, i+1)
        sns.heatmap(np.abs(weight[:50,:50]), cmap='viridis', 
                   cbar=False if i!=len(weights)-1 else True)
        plt.title(f'{name} (abs)', fontsize=10)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

visualize_features(model, test_loader)

# 在训练前添加检查点恢复
checkpoint_path = 'model_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"从检查点恢复训练，epoch {start_epoch}")
else:
    start_epoch = 0

# 在训练循环中添加检查点保存
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss[-1],
    }, checkpoint_path)
visualize_features(model, valid_loader)
visualize_features(model, test_loader)


# 修改早停策略参数
early_stopper = EarlyStopping(
    patience=20,  # 增加耐心值，从15增加到20
    delta=0.002   # 放宽判断阈值，从0.001增加到0.002
)

# 修改学习率调度器参数
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=8,  # 从5增加到8
    min_lr=1e-5  # 添加最小学习率限制
)

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# 在训练循环结束后添加
class_names = [f'Fault_{i}' for i in range(num_classes)]

# 混淆矩阵
plot_confusion_matrix(all_targets, all_preds, class_names, 'confusion_matrix.pdf')

# 高级特征可视化
visualize_features_advanced(model, test_loader, 'feature_analysis.pdf')

# 权重可视化
visualize_weights(model, 'weight_visualization.pdf')

# 分类报告
report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv('classification_report.csv')