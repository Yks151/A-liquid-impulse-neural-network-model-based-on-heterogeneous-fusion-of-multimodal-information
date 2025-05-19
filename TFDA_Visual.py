import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import GradCAM, GradCAMpp
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
from Convmmamba import ConvNetWithMambaAndSelfAttention
from utils import read_directory
class ModelVisualizer:
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        # 修改颜色列表，确保有足够多的颜色
        self.colors = plt.cm.tab20.colors[:num_classes] if num_classes <= 20 else plt.cm.tab20.colors * (num_classes // 20 + 1)[:num_classes]
        self.font_props = {'fontsize': 10, 'fontweight': 'normal'}
        self.title_props = {'fontsize': 12, 'fontweight': 'bold'}
        plt.style.use('seaborn-v0_8')
        self.nature_style = {
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 600,
            'savefig.dpi': 600,
            'pdf.fonttype': 42,
            'font.family': 'Arial',
            'axes.grid': True,
            'grid.alpha': 0.3
        }
    
    def plot_activation_maps(self, test_features, layer_name='net1'):
        """可视化指定层的激活图"""
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 注册hook
        for name, layer in self.model.named_modules():
            if name == layer_name:
                layer.register_forward_hook(get_activation(name))
                break
        
        # 前向传播获取激活
        with torch.no_grad():
            _ = self.model(test_features[:1].to(self.device), 
                         torch.randn(1, 1, 433).to(self.device))
        
        # 可视化激活图
        act = activation[layer_name].squeeze().cpu().numpy()
        fig = plt.figure(figsize=(12, 6), dpi=600)
        
        if len(act.shape) == 4:  # 卷积层激活
            for i in range(min(16, act.shape[1])):  # 最多显示16个通道
                ax = fig.add_subplot(4, 4, i+1)
                ax.imshow(act[0, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i}')
        else:  # 其他层激活
            plt.plot(act.flatten())
            plt.title('Activation Values')
        
        plt.tight_layout()
        return fig

    def overlay_mask(self, img, mask, alpha=0.5):
        """叠加原始图像和热力图"""
        # 将热力图转换为RGB
        mask = np.array(mask)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        
        # 应用颜色映射
        mask = mask.convert('L')
        mask = np.array(mask)
        mask = plt.cm.jet(mask)[..., :3] * 255
        mask = Image.fromarray(mask.astype(np.uint8))
        
        # 调整大小以匹配原始图像
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.BILINEAR)
            
        # 叠加图像
        return Image.blend(img, mask, alpha)

    def plot_gradcam_plusplus(self, test_features, layer_name=None):
        """简化版Grad-CAM++可视化"""
        if layer_name is None:
            for name, layer in self.model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    layer_name = name
                    print(f"Using Conv2d layer: {layer_name}")
            
            if layer_name is None:
                raise ValueError("No convolutional layer found in the model")
        
        # 初始化GradCAM++
        cam_extractor = GradCAMpp(self.model, layer_name)
        
        # 获取第一个样本的激活图
        input_tensor = test_features[:1].to(self.device)
        input_tensor.requires_grad_(True)
        dummy_input2 = torch.randn(1, 1, 433).to(self.device)
        
        # 前向传播
        with torch.set_grad_enabled(True):
            out = self.model(input_tensor, dummy_input2)
            activation_map = cam_extractor(out[0].argmax().item(), out[0])
            cam_extractor.remove_hooks()
        
        # 生成热力图
        heatmap = activation_map[0].squeeze().cpu().numpy()
        
        # 生成叠加图像
        overlay = self.overlay_mask(
            Image.fromarray((test_features[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)),
            Image.fromarray((heatmap*255).astype(np.uint8)), 
            alpha=0.5
        )
        
        # 可视化 - 简化布局
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
        
        # 原始图像
        original_img = test_features[0].permute(1, 2, 0).cpu().numpy()
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Grad-CAM++热力图
        heatmap = activation_map[0].squeeze().cpu().numpy()
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Heatmap')
        ax2.axis('off')
        
        # 叠加效果
        overlay = self.overlay_mask(
            Image.fromarray((original_img*255).astype(np.uint8)),
            Image.fromarray((heatmap*255).astype(np.uint8)), 
            alpha=0.5
        )
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        return fig

    def plot_feature_attention(self, test_features, test_features2):
        """Enhanced time-frequency feature attention visualization"""
        fig = plt.figure(figsize=(20, 12), dpi=600)
        
        # 1. 时域信号(带关注度颜色映射)
        ax1 = fig.add_subplot(231)
        signal = test_features2[0,0,:].numpy()
        t = np.arange(len(signal))
        
        # 获取模型关注权重
        with torch.no_grad():
            _, _, _, attn_weights, _ = self.model(
                test_features[:1].to(self.device),
                test_features2[:1].to(self.device)
            )
            attn = attn_weights[0].cpu().numpy()
            time_attn = attn.mean(0) if len(attn.shape) > 1 else attn
        
        # 使用红到紫的渐变色表示关注度
        cmap = plt.cm.RdPu
        norm = plt.Normalize(time_attn.min(), time_attn.max())
        
        for i in range(len(signal)-1):
            ax1.plot([t[i], t[i+1]], [signal[i], signal[i+1]], 
                    color=cmap(norm(time_attn[i])),
                    linewidth=1.5, alpha=0.8)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, label='Attention Level (Red=High, Purple=Low)')
        
        # 2. 频域信号(带关注度颜色映射)
        ax2 = fig.add_subplot(232)
        signal = test_features2[0,0,:].numpy()
        t = np.arange(len(signal))
        
        # 获取模型关注权重
        with torch.no_grad():
            _, _, _, attn_weights, _ = self.model(
                test_features[:1].to(self.device),
                test_features2[:1].to(self.device)
            )
            attn = attn_weights[0].cpu().numpy()
            time_attn = attn.mean(0) if len(attn.shape) > 1 else attn
        
        # 使用红到紫的渐变色表示关注度
        cmap = plt.cm.RdPu
        norm = plt.Normalize(time_attn.min(), time_attn.max())
        
        for i in range(len(signal)-1):
            ax2.plot([t[i], t[i+1]], [signal[i], signal[i+1]], 
                    color=cmap(norm(time_attn[i])),
                    linewidth=1.5, alpha=0.8)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax2, label='Attention Level (Red=High, Purple=Low)')
        
        plt.tight_layout()
        return fig

    def plot_decision_boundary(self, test_features, test_features2, test_labels):
        """优化版决策边界可视化"""
        with torch.no_grad():
            # 确保输入数据维度正确
            test_features = test_features.to(self.device)
            test_features2 = test_features2.to(self.device)
            
            # 获取模型输出，只获取融合后的特征
            outputs = self.model(test_features, test_features2)
            features = outputs[-1]  # 获取最后一个输出（融合特征）
            features = features.cpu().numpy()
        
        # 使用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)
        
        fig = plt.figure(figsize=(10, 8), dpi=300)
        
        # 绘制散点图
        for i in range(self.num_classes):
            mask = (test_labels == i)
            plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                       color=self.colors[i], 
                       label=f'Class {i}', 
                       s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
        
        # 添加决策边界
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', gamma='scale')
        clf.fit(reduced, test_labels)
        
        # 创建网格
        x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
        y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # 预测网格点
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.1, levels=self.num_classes-1, 
                    colors=self.colors[:self.num_classes])
        
        plt.title('Feature Space Decision Boundary', fontweight='bold', pad=15)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_tsne_visualization(self, features, labels, title='3D t-SNE Visualization'):
        """优化版3D t-SNE降维可视化"""
        tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
        reduced = tsne.fit_transform(features)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为每个类别绘制散点
        for i in range(self.num_classes):
            mask = (labels == i)
            ax.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                     color=self.colors[i], 
                     label=f'Class {i}',
                     s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
        
        # 优化视觉效果
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('t-SNE 1', fontsize=12, labelpad=10)
        ax.set_ylabel('t-SNE 2', fontsize=12, labelpad=10)
        ax.set_zlabel('t-SNE 3', fontsize=12, labelpad=10)
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)  # 调整为更清晰的视角
        ax.set_facecolor('#f0f0f0')  # 设置浅色背景提高对比度
        
        plt.tight_layout()
        return fig
    def plot_3d_confusion_matrix(self, y_true, y_pred):
        """优化版3D混淆矩阵可视化"""
        # 获取所有唯一类别标签
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        num_classes = len(classes)

        # 生成混淆矩阵（注意：这里交换了y_true和y_pred的位置）
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 使用双色渐变方案
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_classes))

        # 三维网格布局（x轴为真实类别，y轴为预测类别）
        xpos, ypos = np.meshgrid(np.arange(num_classes), np.arange(num_classes))
        xpos = xpos.flatten() + 0.5  # 真实类别（x轴）
        ypos = ypos.flatten() + 0.5  # 预测类别（y轴）
        zpos = np.zeros_like(xpos)
        dx = dy = 0.8 * np.ones_like(zpos)
        dz = cm_normalized.flatten()  # 正则化精度（z轴）

        # 按高度排序避免遮挡
        sort_idx = np.argsort(dz)
        colors = colors[sort_idx % num_classes]

        # 绘制立方体
        ax.bar3d(xpos[sort_idx], ypos[sort_idx], zpos[sort_idx],
                dx[sort_idx], dy[sort_idx], dz[sort_idx],
                color=colors, alpha=0.9, edgecolor='k', linewidth=0.5)

        # 添加数值标注（红色字体白色半透明背景）
        for i in range(num_classes):
            for j in range(num_classes):
                height = cm_normalized[i,j]
                if height > 0.01:  # 只标注大于1%的值
                    ax.text(j+0.5, i+0.5, height+0.02, 
                           f"{height:.2f}",
                           ha='center', va='center', fontsize=9,
                           color='red',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 坐标轴标签设置
        ax.set_xticks(np.arange(num_classes)+0.5)
        ax.set_yticks(np.arange(num_classes)+0.5)
        ax.set_xticklabels([f'True {cls}' for cls in classes],  # x轴标签
                          fontsize=9, rotation=35, ha='right')
        ax.set_yticklabels([f'Pred {cls}' for cls in classes],  # y轴标签
                          fontsize=9, rotation=-35, ha='left')
        ax.set_zlabel('Normalized Accuracy', fontsize=11, labelpad=12)  # z轴标签
        ax.set_title('3D Confusion Matrix', fontsize=14, pad=20)

        # 视角优化
        ax.view_init(elev=25, azim=135)
        ax.set_box_aspect((1,1,0.7))
        ax.grid(False)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10, label='Class Distribution')

        plt.tight_layout()
        return fig

    def plot_performance_violin(self, y_true, y_pred, metrics=None):
        """改进的多指标分类性能小提琴图"""
        fig = plt.figure(figsize=(18, 6))
        
        # 默认指标计算方式
        if metrics is None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics = {
                'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
                'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
                'F1-Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
        
        # 计算各项指标
        results = {}
        for name, func in metrics.items():
            results[name] = []
            for cls in range(self.num_classes):
                mask = (y_true == cls)
                if mask.any():
                    results[name].append(func(y_true[mask], y_pred[mask]))
                else:
                    results[name].append(0)  # 用0代替NaN
        
        # 绘制三个子图
        for idx, (name, values) in enumerate(results.items()):
            ax = fig.add_subplot(1, 3, idx+1)
            
            # 确保数据格式正确
            values = np.array(values).reshape(-1, 1)  # 转换为2D数组
            positions = np.arange(1, self.num_classes+1)
            
            # 绘制小提琴图
            vp = ax.violinplot(dataset=values.T, positions=positions,
                             showmeans=True, showmedians=True)
            
            # 设置颜色和样式
            for pc in vp['bodies']:
                pc.set_facecolor(self.colors[idx])
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            ax.set_title(f'{name} by Class', fontweight='bold', pad=15)
            ax.set_xticks(positions)
            ax.set_xticklabels([f'Class {i}' for i in range(self.num_classes)], 
                             rotation=45, fontsize=10)
            ax.set_ylabel(name, fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # 标注数值
            for i, val in enumerate(values):
                ax.text(i+1, val[0]+0.02, f"{val[0]:.2f}", 
                       ha='center', fontsize=10, color='black')

        plt.tight_layout()
        return fig

    def plot_fft_interpretability(self, test_features2, test_labels, class_idx):
        """改进的FFT频谱可解释性分析"""
        with plt.style.context(self.nature_style):
            # 获取该类所有样本
            class_samples = test_features2[test_labels == class_idx]
            avg_signal = class_samples.mean(0)[0].numpy()
            
            # 获取注意力权重
            with torch.no_grad():
                _, _, _, attn_weights, _ = self.model(
                    torch.randn(len(class_samples), 3, 64, 64).to(self.device),
                    class_samples.to(self.device)
                )
                attn = attn_weights.mean(0).cpu().numpy()
            
            fig = plt.figure(figsize=(15, 6))
            
            # FFT频谱
            ax1 = fig.add_subplot(121)
            fft = np.abs(np.fft.fft(avg_signal))
            freq = np.fft.fftfreq(len(avg_signal))
            
            # 使用颜色映射表示关注度
            norm = plt.Normalize(attn.min(), attn.max())
            cmap = plt.cm.viridis
            
            for i in range(len(freq)//2):
                ax1.plot([freq[i], freq[i+1]], [fft[i], fft[i+1]],
                        color=cmap(norm(attn[i])), linewidth=2)
            
            # 标记高关注区域
            top_idx = np.argsort(attn)[-3:]
            for idx in top_idx:
                if idx < len(freq)//2:
                    # 添加红色圆圈标注重要频率区域
                    circle = plt.Circle((freq[idx], fft[idx]), 0.05, 
                                      color='red', fill=False, linewidth=2)
                    ax1.add_patch(circle)
                    ax1.annotate(f'关键频率: {freq[idx]:.2f}Hz\n关注度: {attn[idx]:.2f}',
                               xy=(freq[idx], fft[idx]),
                               xytext=(freq[idx], fft[idx]*1.3),
                               arrowprops=dict(facecolor='red', shrink=0.05),
                               fontsize=10, color='red',
                               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            
            ax1.set_title(f'Class {class_idx} FFT Spectrum with Attention', fontweight='bold')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Magnitude')
            
            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax1, label='Attention Level')
            
            # 注意力权重分布
            ax2 = fig.add_subplot(122)
            ax2.plot(attn, color='#ff7f0e')
            ax2.set_title('Attention Weights Distribution', fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Attention Weight')
            
            plt.tight_layout()
            return fig

    def generate_report(self, test_features, test_features2, test_labels):
        with PdfPages('Report.pdf') as pdf:
            plt.rcParams.update(self.nature_style)
            
            # 1. 决策边界可视化
            fig1 = self.plot_decision_boundary(test_features, test_features2, test_labels)
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close()
            
            # 2. t-SNE可视化
            with torch.no_grad():
                _, _, _, _, features = self.model(
                    test_features.to(self.device),
                    test_features2.to(self.device)
                )
                features = features.cpu().numpy()
            
            fig2 = self.plot_tsne_visualization(features, test_labels.numpy())
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close()
            
            # 3. 3D混淆矩阵
            with torch.no_grad():
                _, test_probas, _, _, _ = self.model(
                    test_features.to(self.device),
                    test_features2.to(self.device)
                )
                test_pred = test_probas.argmax(1).cpu().numpy()
            
            # 3D归一化混淆矩阵
            fig3 = self.plot_3d_confusion_matrix(test_labels.numpy(), test_pred)
            pdf.savefig(fig3, bbox_inches='tight')
            plt.close()
            
            # 多指标小提琴图
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'Accuracy': accuracy_score,
                'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
                'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
                'F1-Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
            }
            # 修改调用方式
            fig5 = self.plot_performance_violin(test_labels.numpy(), test_pred)
            pdf.savefig(fig5, bbox_inches='tight')
            plt.close()
            
            # 3. FFT interpretability for each class
            for i in range(self.num_classes):
                fig4 = self.plot_fft_interpretability(test_features2, test_labels, i)
                pdf.savefig(fig4, bbox_inches='tight')
                plt.close()
            
            # 4. 每类SP图Grad-CAM++
            for i in range(self.num_classes):
                class_samples = test_features[test_labels == i]
                if len(class_samples) > 0:
                    fig6 = self.plot_gradcam_plusplus(class_samples[:1])
                    pdf.savefig(fig6, bbox_inches='tight')
                    plt.close()
            
            # 5. 性能指标小提琴图
            from sklearn.metrics import precision_score, recall_score, f1_score
            y_true = test_labels.numpy()
            y_pred = test_pred           
            metrics = {
                'Precision': precision_score(y_true, y_pred, average=None),
                'Recall': recall_score(y_true, y_pred, average=None),
                'F1-Score': f1_score(y_true, y_pred, average=None)
            }            
            fig7 = plt.figure(figsize=(15, 5))
            for idx, (name, values) in enumerate(metrics.items()):
                ax = fig7.add_subplot(1, 3, idx+1)
                vp = ax.violinplot(dataset=[values], showmeans=True, showmedians=True)
                for pc in vp['bodies']:
                    pc.set_facecolor(self.colors[idx])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                ax.set_title(f'{name} Distribution', fontweight='bold')
                ax.set_ylabel(name)
                for i, val in enumerate(values):
                    ax.text(i+1, val+0.02, f'{val:.2f}', ha='center', fontsize=10)           
            pdf.savefig(fig7, bbox_inches='tight')
            plt.close()
def main():
    # 示例使用代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # 修改模型加载方式
    try:
        # 首先尝试安全模式加载
        model = torch.load('model/best_lnn_snn.pth', map_location=device, weights_only=True)
    except:
        # 如果失败，使用传统方式加载（仅当您信任模型来源时使用）
        model = torch.load('model/best_lnn_snn.pth', map_location=device, weights_only=False)
    model.eval() 
    # 创建可视化工具实例
    visualizer = ModelVisualizer(model, device, num_classes=7)  # 假设有5类
    # 加载测试数据
    x_test, y_test = read_directory('小波时频1/200-7类/test_img', height=64, width=64, normal=1)
    datafft = loadmat('FFT频谱1/FFT.mat')
    ss2 = StandardScaler()  # 初始化标准化器
    x_test2 = datafft['test_X']
    
    # 数据预处理 - 先拟合再转换
    ss2.fit(x_test2)  # 添加这行代码
    
    # 数据预处理 - 确保输入维度匹配
    x_test2 = ss2.transform(x_test2)
    x_test2 = x_test2.reshape(x_test2.shape[0], 1, 433)  # 明确指定信号长度为433
    
    # 确保图像和FFT数据样本数量一致
    min_samples = min(len(x_test), len(x_test2))
    x_test = x_test[:min_samples]
    x_test2 = x_test2[:min_samples]
    y_test = y_test[:min_samples]
    
    # 打印输入维度进行检查
    print(f"图像数据维度: {x_test.shape}")
    print(f"FFT数据维度: {x_test2.shape}")
    
    test_features = torch.tensor(x_test).type(torch.FloatTensor)
    test_features2 = torch.tensor(x_test2).type(torch.FloatTensor)
    test_labels = torch.tensor(y_test-1).type(torch.LongTensor)
    
    # 生成完整报告 (修改调用方式)
    visualizer.generate_report(
        test_features=test_features,
        test_features2=test_features2,
        test_labels=test_labels
    )
    print("可视化报告已生成: Nature_Style_Report.pdf")
if __name__ == "__main__":
    main()
