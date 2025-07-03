import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    """
    MovieLens数据集类，用于PyTorch的DataLoader
    """
    
    def __init__(self, data_df):
        """
        初始化数据集
        
        参数:
            data_df: 包含所有特征和标签的DataFrame
        """
        self.data = data_df
        self.user_features = {
            'UserID': torch.LongTensor(data_df['UserID_Idx'].values),
            'Gender': torch.LongTensor(data_df['Gender_Idx'].values),
            'Age': torch.LongTensor(data_df['Age_Idx'].values),
            'Occupation': torch.LongTensor(data_df['Occupation_Idx'].values)
        }
        
        
        # 处理电影类型特征（多值类别）
        # 将字符串格式的索引列表转换为实际的列表
        # 先将字符串转为list[int]，再转为LongTensor
        self.item_features = {
            'MovieID': torch.LongTensor(data_df['MovieID_Idx'].values),
            'Genres': [
                torch.LongTensor(genres_idx)
                for genres_idx in data_df['Genres_Idx'].values
            ]
        }
        self.labels = torch.FloatTensor(data_df['Label'].values)
    
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
    user_features = {
        'UserID': torch.stack([sample[0]['UserID'] for sample in batch]),
        'Gender': torch.stack([sample[0]['Gender'] for sample in batch]),
        'Age': torch.stack([sample[0]['Age'] for sample in batch]),
        'Occupation': torch.stack([sample[0]['Occupation'] for sample in batch])
    }
    
    item_features = {
        'MovieID': torch.stack([sample[1]['MovieID'] for sample in batch]),
        'Genres': [sample[1]['Genres'] for sample in batch]
    }
    
    labels = torch.stack([sample[2] for sample in batch])
    
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
    train_data = pd.read_pickle(os.path.join(data_dir, "train_data.pkl"))
    test_data = pd.read_pickle(os.path.join(data_dir, "test_data.pkl"))
    
    # 加载特征信息
    import pickle
    with open(os.path.join(data_dir, "feature_info.pkl"), "rb") as f:
        feature_info = pickle.load(f)
    
    # 创建数据集
    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_loader, feature_info 