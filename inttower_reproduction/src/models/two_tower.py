import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    标准双塔模型实现
    
    特点:
    - 用户塔和物品塔分别处理各自特征
    - 用户和物品的最终表示通过点积计算相似度
    - 模型输出是交互的预测概率
    """
    
    def __init__(
        self,
        user_feature_dims,
        item_feature_dims,
        embedding_dim=32,
        mlp_dims=[256, 128, 128],
        dropout=0.2
    ):
        """
        初始化双塔模型
        
        参数:
            user_feature_dims: dict, 用户特征的维度字典 {特征名: 类别数}
            item_feature_dims: dict, 物品特征的维度字典 {特征名: 类别数}
            embedding_dim: int, 嵌入向量的维度
            mlp_dims: list, MLP各层的维度
            dropout: float, Dropout比率
        """
        super(TwoTowerModel, self).__init__()
        
        # 用户特征嵌入
        self.user_embeddings = nn.ModuleDict({
            feature: nn.Embedding(dim, embedding_dim)
            for feature, dim in user_feature_dims.items()
        })
        
        # 物品特征嵌入
        self.item_embeddings = nn.ModuleDict()
        for feature, dim in item_feature_dims.items():
            if feature == 'Genres':
                # 使用EmbeddingBag处理多值类别特征
                self.item_embeddings[feature] = nn.EmbeddingBag(
                    dim, embedding_dim, mode='mean'
                )
            else:
                self.item_embeddings[feature] = nn.Embedding(dim, embedding_dim)
        
        # 计算特征数量
        self.num_user_features = len(user_feature_dims)
        self.num_item_features = len(item_feature_dims)
        
        # 计算输入维度
        user_input_dim = self.num_user_features * embedding_dim
        item_input_dim = self.num_item_features * embedding_dim
        
        # 用户塔MLP
        self.user_mlp = self._create_mlp(user_input_dim, mlp_dims, dropout)
        
        # 物品塔MLP
        self.item_mlp = self._create_mlp(item_input_dim, mlp_dims, dropout)
        
        # 存储每一层的输出，用于绘制特征激活图或进一步分析
        self.user_tower_outputs = []
        self.item_tower_outputs = []
        
    def _create_mlp(self, input_dim, mlp_dims, dropout):
        """创建多层感知机"""
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, mlp_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 中间层
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, user_features, item_features, return_embeddings=False):
        """
        前向传播
        
        参数:
            user_features: dict, 用户特征字典 {特征名: 特征值}
            item_features: dict, 物品特征字典 {特征名: 特征值}
            return_embeddings: bool, 是否返回嵌入向量
            
        返回:
            y_pred: 预测分数
            (可选) user_emb, item_emb: 用户和物品的最终嵌入向量
        """
        # 清空之前的塔输出
        self.user_tower_outputs = []
        self.item_tower_outputs = []
        
        # 处理用户特征
        user_embs = []
        for feature, embedding_layer in self.user_embeddings.items():
            user_embs.append(embedding_layer(user_features[feature]))
        
        # 拼接所有用户特征嵌入
        user_emb = torch.cat(user_embs, dim=1)
        
        # 处理物品特征
        item_embs = []
        for feature, embedding_layer in self.item_embeddings.items():
            if feature == 'Genres':
                # 处理电影类型的多值特征
                # offsets是每个样本在展平后的indices中的起始位置
                offsets = torch.zeros(item_features[feature].shape[0], dtype=torch.long, device=item_features[feature].device)
                for i in range(1, item_features[feature].shape[0]):
                    offsets[i] = offsets[i-1] + len(item_features[feature][i-1])
                
                # 将所有样本的类型索引展平为一维
                indices_flattened = torch.cat(item_features[feature])
                
                # 使用EmbeddingBag获取聚合后的嵌入
                genre_emb = self.item_embeddings[feature](
                    indices_flattened, offsets
                )
                item_embs.append(genre_emb)
            else:
                item_embs.append(embedding_layer(item_features[feature]))
        
        # 拼接所有物品特征嵌入
        item_emb = torch.cat(item_embs, dim=1)
        
        # 通过用户塔和物品塔
        for i, layer in enumerate(self.user_mlp):
            user_emb = layer(user_emb)
            if i % 3 == 2:  # 每个完整块(Linear+ReLU+Dropout)后保存输出
                self.user_tower_outputs.append(user_emb)
        
        for i, layer in enumerate(self.item_mlp):
            item_emb = layer(item_emb)
            if i % 3 == 2:  # 每个完整块(Linear+ReLU+Dropout)后保存输出
                self.item_tower_outputs.append(item_emb)
        
        # L2归一化
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        
        # 点积计算预测分数
        y_pred = torch.sum(user_emb * item_emb, dim=1)
        
        if return_embeddings:
            return y_pred, user_emb, item_emb
        else:
            return y_pred 