import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json


class MovieLensDataset(Dataset):
    """
    MovieLens数据集类，用于PyTorch的DataLoader
    """
    
    def __init__(self, data_df, device=None):
        """
        初始化数据集
        
        参数:
            data_df: 包含所有特征和标签的DataFrame
            device: 指定设备(GPU/CPU)，如果为None则默认不将数据移到特定设备
        """
        self.data = data_df
        self.device = device
        
        # 预先将数据转换为张量并固定内存布局
        self.user_features = {
            'UserID': torch.LongTensor(data_df['UserID_Idx'].values).contiguous(),
            'Gender': torch.LongTensor(data_df['Gender_Idx'].values).contiguous(),
            'Age': torch.LongTensor(data_df['Age_Idx'].values).contiguous(),
            'Occupation': torch.LongTensor(data_df['Occupation_Idx'].values).contiguous()
        }
        
        # 如果指定了设备，将数据直接移到设备上
        if self.device is not None:
            self.user_features = {k: v.to(self.device, non_blocking=True) for k, v in self.user_features.items()}
        
        # 处理电影类型特征（多值类别）
        # 将字符串格式的索引列表转换为实际的列表
        # 先将字符串转为list[int]，再转为LongTensor
        movie_ids = torch.LongTensor(data_df['MovieID_Idx'].values).contiguous()
        if self.device is not None:
            movie_ids = movie_ids.to(self.device, non_blocking=True)
            
        self.item_features = {
            'MovieID': movie_ids,
            'Genres': [
                torch.LongTensor(genres_idx).contiguous()
                for genres_idx in data_df['Genres_Idx'].values
            ]
        }
        
        # Genres暂时不移到GPU，因为它是变长序列，会在collate_fn中处理
        
        # 将标签转换为张量并固定内存布局
        self.labels = torch.FloatTensor(data_df['Label'].values).contiguous()
        if self.device is not None:
            self.labels = self.labels.to(self.device, non_blocking=True)
    
    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        user_feat = {k: v[idx] for k, v in self.user_features.items()}
        
        item_feat = {
            'MovieID': self.item_features['MovieID'][idx],
            'Genres': self.item_features['Genres'][idx]
        }
        
        label = self.labels[idx]
        
        return user_feat, item_feat, label


def collate_fn(batch):
    """
    自定义批次处理函数，处理变长序列（电影类型）
    
    参数:
        batch: 一批样本，每个样本是(user_feat, item_feat, label)的元组
        
    返回:
        user_features: 用户特征字典，每个特征形状为[batch_size]
        item_features: 物品特征字典，其中Genres是列表的列表
        labels: 标签张量，形状为[batch_size]
    """
    # 确定设备
    device = None
    if isinstance(batch[0][0]['UserID'], torch.Tensor):
        device = batch[0][0]['UserID'].device
        
    user_features = {
        'UserID': torch.stack([sample[0]['UserID'] for sample in batch]).contiguous(),
        'Gender': torch.stack([sample[0]['Gender'] for sample in batch]).contiguous(),
        'Age': torch.stack([sample[0]['Age'] for sample in batch]).contiguous(),
        'Occupation': torch.stack([sample[0]['Occupation'] for sample in batch]).contiguous()
    }
    
    item_features = {
        'MovieID': torch.stack([sample[1]['MovieID'] for sample in batch]).contiguous(),
        'Genres': [sample[1]['Genres'] for sample in batch]
    }
    
    # 如果Genres不在同一设备上，我们将它移到正确的设备
    if device is not None and item_features['Genres'][0].device != device:
        item_features['Genres'] = [g.to(device, non_blocking=True) for g in item_features['Genres']]
    
    labels = torch.stack([sample[2] for sample in batch]).contiguous()
    
    return user_features, item_features, labels


def get_data_loaders(data_dir="./data/ml-1m/processed", batch_size=2048, num_workers=4):
    """
    创建训练集和测试集的数据加载器
    
    参数:
        data_dir: 处理后数据的目录
        batch_size: 批次大小
        num_workers: 数据加载的并行工作进程数
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        feature_info: 特征信息字典
    """
    # 加载处理后的数据
    train_data = pd.read_parquet(os.path.join(data_dir, "train_data.parquet"))
    test_data = pd.read_parquet(os.path.join(data_dir, "test_data.parquet"))
    
    # 加载特征信息
    with open(os.path.join(data_dir, "feature_info.json"), "r", encoding="utf-8") as f:
        feature_info = json.load(f)
    
    # 决定是否使用CUDA，并为CUDA优化数据加载
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # 使用GPU预热和内存固定
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # 创建数据集
    train_dataset = MovieLensDataset(train_data, device=None)  # 在CPU上创建，通过DataLoader的pin_memory将数据高效移至GPU
    test_dataset = MovieLensDataset(test_data, device=None)
    
    # 优化数据加载器配置
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'pin_memory': use_cuda,  # 使用内存固定加速数据传输到GPU
        'pin_memory_device': str(device) if use_cuda else "",
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=num_workers,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=num_workers,
        **loader_kwargs
    )
    
    return train_loader, test_loader, feature_info 