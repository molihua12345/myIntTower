import torch
import torch.nn as nn
import torch.nn.functional as F
from .two_tower import TwoTowerModel
import numpy as np


class LightSE(nn.Module):
    """
    Light-SE (轻量级Squeeze-and-Excitation)模块
    
    对特征嵌入进行自适应加权，强调不同特征的相对重要性
    与标准SENET的区别：使用Softmax而不是Sigmoid进行激活，确保特征权重之和为1
    """
    
    def __init__(self, num_features):
        """
        初始化Light-SE模块
        
        参数:
            num_features: 特征的数量
        """
        super(LightSE, self).__init__()
        self.fc = nn.Linear(num_features, num_features)
    
    def forward(self, embeds):
        """
        前向传播
        
        参数:
            embeds: 特征嵌入, 形状为 [batch_size, num_features, embed_dim]
            
        返回:
            weighted_embeds: 加权后的特征嵌入
        """
        # 1. Squeeze: 对嵌入向量维度进行均值池化
        # [batch_size, num_features, embed_dim] -> [batch_size, num_features]
        z = torch.mean(embeds, dim=2)
        
        # 2. Excite: 通过一个全连接层并使用Softmax激活
        # [batch_size, num_features] -> [batch_size, num_features]
        weights = self.fc(z)
        weights = F.softmax(weights, dim=1)
        
        # 3. Re-weight: 将权重应用于原始嵌入
        # [batch_size, num_features, 1] * [batch_size, num_features, embed_dim]
        weights = weights.unsqueeze(2)  # [batch_size, num_features, 1]
        weighted_embeds = embeds * weights
        
        return weighted_embeds


class SENET(nn.Module):
    """
    标准SENET模块（用于消融实验）
    
    与Light-SE的区别：使用两层全连接网络，中间有ReLU激活，最后使用Sigmoid激活
    """
    
    def __init__(self, num_features):
        """
        初始化SENET模块
        
        参数:
            num_features: 特征的数量
        """
        super(SENET, self).__init__()
        
        # 两层全连接网络
        self.fc1 = nn.Linear(num_features, num_features // 2)  # 降维
        self.fc2 = nn.Linear(num_features // 2, num_features)  # 恢复维度
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, embeds):
        """
        前向传播
        
        参数:
            embeds: 特征嵌入, 形状为 [batch_size, num_features, embed_dim]
            
        返回:
            weighted_embeds: 加权后的特征嵌入
        """
        # 1. Squeeze: 对嵌入向量维度进行均值池化
        z = torch.mean(embeds, dim=2)
        
        # 2. Excite: 通过两层全连接网络
        weights = self.fc1(z)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)
        
        # 3. Re-weight: 将权重应用于原始嵌入
        weights = weights.unsqueeze(2)
        weighted_embeds = embeds * weights
        
        return weighted_embeds


class FEBlock(nn.Module):
    """
    FE-Block (细粒度早期特征交互)模块
    
    通过用户塔中各层表示与物品的最终表示间的细粒度交互，增强模型表达能力
    同时保持双塔模型的高效率优势
    """
    
    def __init__(self, user_tower_dims, head_num=5, head_dim=64):
        """
        初始化FE-Block模块
        
        参数:
            user_tower_dims: 用户塔各层的维度列表
            head_num: 注意力头的数量（默认为5，等于用户特征数量）
            head_dim: 每个头的维度
        """
        super(FEBlock, self).__init__()
        
        self.head_num = head_num
        self.head_dim = head_dim
        self.num_user_layers = len(user_tower_dims)
        
        # 为用户塔的每一层创建投影层
        self.user_projections = nn.ModuleList([
            nn.Linear(dim, head_num * head_dim)
            for dim in user_tower_dims
        ])
        
        # 为物品塔的最后一层创建投影层
        self.item_projection = nn.Linear(user_tower_dims[-1], head_num * head_dim)
    
    def forward(self, user_tower_outputs, item_final_output):
        """
        前向传播
        
        参数:
            user_tower_outputs: 用户塔各层的输出，列表形式
            item_final_output: 物品塔最后一层的输出
            
        返回:
            interaction_score: 交互分数
        """
        batch_size = user_tower_outputs[0].shape[0]
        item_proj = self.item_projection(item_final_output)
        
        # 将物品投影重塑为 [batch_size, head_num, head_dim]
        item_proj = item_proj.view(batch_size, self.head_num, self.head_dim)
        
        # 计算用户塔每一层与物品的交互分数
        layer_scores = []
        for i, user_output in enumerate(user_tower_outputs):
            # 将用户表示投影到多头空间
            user_proj = self.user_projections[i](user_output)
            user_proj = user_proj.view(batch_size, self.head_num, self.head_dim)
            
            # 计算用户和物品表示的相似度矩阵
            # [batch_size, head_num, head_dim] @ [batch_size, head_dim, head_num]
            # -> [batch_size, head_num, head_num]
            sim_matrix = torch.bmm(
                user_proj, 
                item_proj.transpose(1, 2)
            )
            
            # 对于每个用户头，找出与所有物品头交互的最大相似度值
            # [batch_size, head_num]
            max_sim, _ = sim_matrix.max(dim=2)
            
            # 在所有头上求和得到该层的交互分数
            # [batch_size]
            layer_score = max_sim.sum(dim=1)
            layer_scores.append(layer_score)
        
        # 累加所有层的分数作为最终交互分数
        interaction_score = torch.stack(layer_scores, dim=0).sum(dim=0)
        
        return interaction_score


class CIR(nn.Module):
    """
    CIR (对比交互正则化)模块
    
    通过对比学习增强表示空间的结构，使相关的用户-物品对靠近，不相关的远离
    """
    
    def __init__(self, temperature=0.07):
        """
        初始化CIR模块
        
        参数:
            temperature: InfoNCE损失中的温度参数
        """
        super(CIR, self).__init__()
        self.temperature = temperature
    
    def forward(self, user_embeds, item_embeds):
        """
        前向传播，计算InfoNCE对比损失
        
        参数:
            user_embeds: 用户嵌入，[batch_size, embed_dim]
            item_embeds: 物品嵌入，[batch_size, embed_dim]
            
        返回:
            loss: 对比损失值
        """
        batch_size = user_embeds.shape[0]
        
        # 计算批次内所有用户-物品对的相似度矩阵
        # [batch_size, embed_dim] @ [embed_dim, batch_size] -> [batch_size, batch_size]
        sim_matrix = torch.mm(user_embeds, item_embeds.t()) / self.temperature
        
        # 对角线上的元素是正样本对的相似度
        positive_samples = torch.diag(sim_matrix)
        
        # 计算每个用户对应的InfoNCE损失
        # 对于每个用户，将其与正样本物品的相似度与所有物品的相似度进行对比
        exp_pos = torch.exp(positive_samples)
        exp_all = torch.exp(sim_matrix).sum(dim=1)
        
        # InfoNCE损失: -log(exp(pos) / sum(exp(all)))
        nce_loss = -torch.log(exp_pos / exp_all)
        
        # 返回批次平均损失
        return nce_loss.mean()


class IntTowerModel(nn.Module):
    """
    IntTower模型实现
    
    整合Light-SE、FE-Block和CIR三个创新组件，增强双塔模型的表达能力
    """
    
    def __init__(
        self,
        user_feature_dims,
        item_feature_dims,
        embedding_dim=32,
        mlp_dims=[256, 128, 128],
        dropout=0.2,
        use_light_se=True,
        use_senet=False,
        use_fe_block=True,
        use_cir=True,
        head_num=5,
        head_dim=64,
        temperature=0.07
    ):
        """
        初始化IntTower模型
        
        参数:
            user_feature_dims: 用户特征维度字典
            item_feature_dims: 物品特征维度字典
            embedding_dim: 嵌入维度
            mlp_dims: MLP各层维度
            dropout: Dropout比率
            use_light_se: 是否使用Light-SE
            use_senet: 是否使用SENET（与Light-SE互斥）
            use_fe_block: 是否使用FE-Block
            use_cir: 是否使用CIR
            head_num: FE-Block的头数
            head_dim: FE-Block的头维度
            temperature: CIR的温度参数
        """
        super(IntTowerModel, self).__init__()
        
        # 基础的双塔模型
        self.two_tower = TwoTowerModel(
            user_feature_dims,
            item_feature_dims,
            embedding_dim,
            mlp_dims,
            dropout
        )
        
        # 特征数量
        self.num_user_features = len(user_feature_dims)
        self.num_item_features = len(item_feature_dims)
        
        # 模块配置
        self.use_light_se = use_light_se
        self.use_senet = use_senet
        self.use_fe_block = use_fe_block
        self.use_cir = use_cir
        
        # 初始化各模块
        if use_light_se:
            self.light_se_user = LightSE(self.num_user_features)
            self.light_se_item = LightSE(self.num_item_features)
        elif use_senet:
            self.senet_user = SENET(self.num_user_features)
            self.senet_item = SENET(self.num_item_features)
            
        if use_fe_block:
            self.fe_block = FEBlock(mlp_dims, head_num, head_dim)
            
        if use_cir:
            self.cir = CIR(temperature)
            
        # 如果使用FC层替代FE-Block
        self.use_fc = not use_fe_block and not use_senet and not use_light_se
        if self.use_fc:
            concat_dim = mlp_dims[-1] * 2
            self.fc_interaction = nn.Linear(concat_dim, 1)
            
    def forward(self, user_features, item_features):
        """
        前向传播
        
        参数:
            user_features: 用户特征字典
            item_features: 物品特征字典
            
        返回:
            y_pred: 预测分数
            loss_cir: CIR损失（如果启用）
        """
        # 清空之前的塔输出，防止累积
        self.two_tower.user_tower_outputs = []
        self.two_tower.item_tower_outputs = []

        # 准备特征嵌入
        user_embs = []
        for feature, embedding_layer in self.two_tower.user_embeddings.items():
            user_embs.append(embedding_layer(user_features[feature]).unsqueeze(1))
            
        item_embs = []
        for feature, embedding_layer in self.two_tower.item_embeddings.items():
            if feature == 'Genres':
                # 处理电影类型的多值特征
                batch_size = len(item_features[feature])
                lengths = [len(x) for x in item_features[feature]]
                offsets = torch.tensor([0] + list(np.cumsum(lengths)[:-1]), dtype=torch.long, device=item_features[feature][0].device)
                indices_flattened = torch.cat(item_features[feature])
                genre_emb = embedding_layer(indices_flattened, offsets)
                item_embs.append(genre_emb.unsqueeze(1))
            else:
                item_embs.append(embedding_layer(item_features[feature]).unsqueeze(1))
        
        # 将嵌入堆叠为 [batch_size, num_features, embed_dim]
        user_embs = torch.cat(user_embs, dim=1)
        item_embs = torch.cat(item_embs, dim=1)
        
        # 应用特征注意力
        if self.use_light_se:
            user_embs = self.light_se_user(user_embs)
            item_embs = self.light_se_item(item_embs)
        elif self.use_senet:
            user_embs = self.senet_user(user_embs)
            item_embs = self.senet_item(item_embs)
            
        # 扁平化嵌入为 [batch_size, num_features * embed_dim]
        batch_size = user_embs.shape[0]
        user_embs = user_embs.view(batch_size, -1)
        item_embs = item_embs.view(batch_size, -1)
        
        # 通过用户塔和物品塔
        for i, layer in enumerate(self.two_tower.user_mlp):
            user_embs = layer(user_embs)
            if isinstance(layer, nn.Linear):
                self.two_tower.user_tower_outputs.append(user_embs)
        
        for i, layer in enumerate(self.two_tower.item_mlp):
            item_embs = layer(item_embs)
            if isinstance(layer, nn.Linear):
                self.two_tower.item_tower_outputs.append(item_embs)
        
        # 获取最终的用户和物品表示
        user_emb_final = self.two_tower.user_tower_outputs[-1]
        item_emb_final = self.two_tower.item_tower_outputs[-1]
        
        # 归一化用于点积交互和CIR损失
        user_emb_norm = F.normalize(user_emb_final, p=2, dim=1)
        item_emb_norm = F.normalize(item_emb_final, p=2, dim=1)
        
        # 选择交互方式
        if self.use_fe_block:
            # 使用FE-Block进行细粒度早期交互
            y_pred = self.fe_block(
                self.two_tower.user_tower_outputs, 
                item_emb_final
            )
        elif self.use_fc:
            # 使用FC层替代FE-Block（消融实验）
            concat_emb = torch.cat([user_emb_final, item_emb_final], dim=1)
            y_pred = self.fc_interaction(concat_emb).squeeze(1)
        else:
            # 回退到标准点积交互
            y_pred = torch.sum(user_emb_norm * item_emb_norm, dim=1)
        
        # 计算CIR损失
        loss_cir = None
        if self.use_cir and self.training:
            loss_cir = self.cir(user_emb_norm, item_emb_norm)
            
        if loss_cir is not None:
            return y_pred, loss_cir
        else:
            return y_pred 