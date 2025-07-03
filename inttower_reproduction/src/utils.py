import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss


class EarlyStopping:
    """早停类，用于防止过拟合"""
    
    def __init__(self, patience=5, min_delta=0, mode='max'):
        """
        初始化早停对象
        
        参数:
            patience: 容忍多少个epoch没有改进
            min_delta: 最小改进阈值
            mode: 'max'表示监控指标越大越好，'min'表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        根据新的验证分数，决定是否早停
        
        参数:
            score: 当前验证分数
            
        返回:
            True表示需要早停，False表示继续训练
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            return True
        return False


def setup_seed(seed=42):
    """设置随机种子，确保实验可重复"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """计算AUC指标"""
    return roc_auc_score(y_true, y_pred)


def calculate_logloss(y_true, y_pred):
    """计算Logloss指标"""
    # 将预测概率限制在[1e-7, 1-1e-7]范围内，防止log(0)错误
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_pred)


def save_model(model, save_dir, filename):
    """保存模型权重"""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, filename))
    print(f"模型已保存至: {os.path.join(save_dir, filename)}")


def load_model(model, model_path):
    """加载模型权重"""
    model.load_state_dict(torch.load(model_path))
    return model


def log_metrics(epoch, train_loss, train_auc, train_logloss, test_auc, test_logloss):
    """记录训练和测试指标"""
    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Logloss: {train_logloss:.4f}")
    print(f"  Test AUC: {test_auc:.4f}, Test Logloss: {test_logloss:.4f}")


class MetricTracker:
    """跟踪和记录训练过程中的指标"""
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_aucs = []
        self.train_loglosses = []
        self.test_aucs = []
        self.test_loglosses = []
        
    def update(self, epoch, train_loss, train_auc, train_logloss, test_auc, test_logloss):
        """更新指标"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_aucs.append(train_auc)
        self.train_loglosses.append(train_logloss)
        self.test_aucs.append(test_auc)
        self.test_loglosses.append(test_logloss)
        
    def save_metrics(self, save_path):
        """保存指标到文件"""
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_aucs': self.train_aucs,
            'train_loglosses': self.train_loglosses,
            'test_aucs': self.test_aucs,
            'test_loglosses': self.test_loglosses
        }
        
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        print(f"指标已保存至: {save_path}")
        
    def plot_metrics(self, save_dir):
        """绘制训练过程中的指标曲线"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "train_loss.png"))
        plt.close()
        
        # 绘制AUC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_aucs, label="Train AUC")
        plt.plot(self.epochs, self.test_aucs, label="Test AUC")
        plt.title("AUC Curves")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "auc_curves.png"))
        plt.close()
        
        # 绘制Logloss曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_loglosses, label="Train Logloss")
        plt.plot(self.epochs, self.test_loglosses, label="Test Logloss")
        plt.title("Logloss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Logloss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "logloss_curves.png"))
        plt.close()


def visualize_ablation_results(results_dict, save_path):
    """
    可视化消融实验结果
    
    参数:
        results_dict: 字典，包含每个模型变体的AUC和Logloss
        save_path: 保存图像的路径
    """
    # 准备数据
    models = list(results_dict.keys())
    aucs = [results_dict[model]['auc'] for model in models]
    loglosses = [results_dict[model]['logloss'] for model in models]
    
    # 设置图形大小和风格
    plt.figure(figsize=(12, 10))
    sns.set_style('whitegrid')
    
    # 绘制AUC子图
    plt.subplot(2, 1, 1)
    bars = plt.bar(models, aucs, color='skyblue')
    plt.title('AUC Comparison', fontsize=14)
    plt.ylabel('AUC', fontsize=12)
    plt.ylim(min(aucs) * 0.95, max(aucs) * 1.05)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # 绘制Logloss子图
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, loglosses, color='salmon')
    plt.title('Logloss Comparison', fontsize=14)
    plt.ylabel('Logloss', fontsize=12)
    plt.ylim(min(loglosses) * 0.95, max(loglosses) * 1.05)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"消融实验结果可视化已保存至: {save_path}") 