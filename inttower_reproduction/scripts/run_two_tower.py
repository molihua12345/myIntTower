"""
Two-Tower基线模型训练脚本
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import TwoTowerModel
from src.data_loader import get_data_loaders
from src.utils import (
    setup_seed, log_metrics, calculate_auc, calculate_logloss,
    MetricTracker, save_model, EarlyStopping
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Two-Tower模型训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./data/ml-1m/processed", 
                        help="预处理后的数据目录")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="训练批次大小")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="数据加载的并行工作进程数")
    
    # 模型参数
    parser.add_argument("--embedding_dim", type=int, default=32, 
                        help="特征嵌入维度")
    parser.add_argument("--mlp_dims", type=str, default="256,128,128", 
                        help="MLP各层的维度，用逗号分隔")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout比率")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=30, 
                        help="训练轮数")
    parser.add_argument("--patience", type=int, default=5, 
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU编号，-1表示使用CPU")
    
    # 输出参数
    parser.add_argument("--save_dir", type=str, default="./results/two_tower", 
                        help="模型和结果保存目录")
    
    return parser.parse_args()


def train(args, model, train_loader, test_loader, device):
    """
    训练Two-Tower模型
    
    参数:
        args: 命令行参数
        model: Two-Tower模型实例
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 训练设备
        
    返回:
        最佳测试集AUC和Logloss
    """
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 设置损失函数
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 设置早停
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # 创建指标追踪器
    metric_tracker = MetricTracker()
    
    # 记录最佳性能
    best_auc = 0
    best_logloss = float('inf')
    best_epoch = -1
    
    # 开始训练
    print(f"开始训练Two-Tower模型，总轮数：{args.epochs}")
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for user_features, item_features, labels in train_iter:
            # 将数据转移到设备
            user_features = {k: v.to(device) for k, v in user_features.items()}
            item_features = {
                'MovieID': item_features['MovieID'].to(device),
                'Genres': [g.to(device) for g in item_features['Genres']]
            }
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            logits = model(user_features, item_features)
            loss = criterion(logits, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 记录损失和预测
            train_loss += loss.item() * labels.size(0)
            train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            
            # 更新进度条
            train_iter.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算训练集指标
        train_loss /= len(train_loader.dataset)
        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)
        train_auc = calculate_auc(train_labels, train_preds)
        train_logloss = calculate_logloss(train_labels, train_preds)
        
        # 测试阶段
        model.eval()
        test_preds = []
        test_labels = []
        
        # 使用tqdm显示进度条
        test_iter = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]")
        
        with torch.no_grad():
            for user_features, item_features, labels in test_iter:
                # 将数据转移到设备
                user_features = {k: v.to(device) for k, v in user_features.items()}
                item_features = {
                    'MovieID': item_features['MovieID'].to(device),
                    'Genres': [g.to(device) for g in item_features['Genres']]
                }
                labels = labels.to(device)
                
                # 前向传播
                logits = model(user_features, item_features)
                
                # 记录预测
                test_preds.append(torch.sigmoid(logits).cpu().numpy())
                test_labels.append(labels.cpu().numpy())
        
        # 计算测试集指标
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        test_auc = calculate_auc(test_labels, test_preds)
        test_logloss = calculate_logloss(test_labels, test_preds)
        
        # 记录指标
        log_metrics(epoch, train_loss, train_auc, train_logloss, test_auc, test_logloss)
        metric_tracker.update(epoch, train_loss, train_auc, train_logloss, test_auc, test_logloss)
        
        # 更新最佳性能
        if test_auc > best_auc:
            best_auc = test_auc
            best_logloss = test_logloss
            best_epoch = epoch
            
            # 保存最佳模型
            save_model(model, args.save_dir, "best_model.pth")
        
        # 早停检查
        if early_stopping(test_auc):
            print(f"早停: {args.patience}轮内未见改善")
            break
    
    print(f"训练完成。最佳性能 (Epoch {best_epoch+1}):")
    print(f"  Test AUC: {best_auc:.4f}")
    print(f"  Test Logloss: {best_logloss:.4f}")
    
    # 保存训练指标
    os.makedirs(args.save_dir, exist_ok=True)
    metric_tracker.save_metrics(os.path.join(args.save_dir, "metrics.pkl"))
    metric_tracker.plot_metrics(args.save_dir)
    
    # 返回最佳性能
    return best_auc, best_logloss


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader, feature_info = get_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)}个样本, 测试集: {len(test_loader.dataset)}个样本")
    
    # 解析MLP维度
    mlp_dims = [int(dim) for dim in args.mlp_dims.split(',')]
    
    # 创建模型
    print("创建Two-Tower模型...")
    model = TwoTowerModel(
        user_feature_dims=feature_info['user_feature_dims'],
        item_feature_dims=feature_info['item_feature_dims'],
        embedding_dim=args.embedding_dim,
        mlp_dims=mlp_dims,
        dropout=args.dropout
    )
    model.to(device)
    
    # 训练模型
    start_time = time.time()
    best_auc, best_logloss = train(args, model, train_loader, test_loader, device)
    total_time = time.time() - start_time
    
    # 保存结果
    results = {
        "model": "Two-Tower",
        "best_auc": best_auc,
        "best_logloss": best_logloss,
        "training_time": total_time
    }
    
    import json
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"结果已保存至: {args.save_dir}")


if __name__ == "__main__":
    main() 