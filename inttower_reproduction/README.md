# IntTower复现项目

本项目复现论文《IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System》中的方法和实验结果。

## 项目简介

IntTower是一种改进的双塔模型，旨在解决传统双塔模型表达能力有限的问题。它通过三个核心创新组件增强了双塔架构：

1. Light-SE: 轻量级Squeeze-and-Excitation注意力机制
2. FE-Block: 细粒度早期特征交互模块
3. CIR: 对比交互正则化

## 项目结构

```
inttower_reproduction/
├── data/                        # 数据目录
│   └── ml-1m/                   # 原始MovieLens-1M数据
│       └── processed/           # 预处理后的数据
├── src/                         # 源代码目录
│   ├── models/                  # 模型实现
│   │   ├── __init__.py          # 模型包初始化
│   │   ├── two_tower.py         # 基线双塔模型
│   │   └── inttower.py          # IntTower模型及其组件
│   ├── preprocessing.py         # 数据预处理
│   ├── data_loader.py           # 数据加载类
│   └── utils.py                 # 工具函数
├── scripts/                     # 脚本目录
│   ├── download_data.py         # 数据下载脚本
│   ├── preprocess_data.py       # 数据预处理脚本
│   ├── run_two_tower.py         # 基线模型训练脚本
│   ├── run_inttower.py          # IntTower模型训练脚本
│   └── visualize_results.py     # 结果可视化脚本
├── results/                     # 实验结果保存目录
├── run_all_experiments.py       # 运行所有实验的主脚本
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明文档
```

## 环境设置

```bash
# 创建虚拟环境
conda create -n inttower python=3.8
conda activate inttower

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

1. 下载MovieLens-1M数据集：
```bash
python scripts/download_data.py
```

2. 数据预处理：
```bash
python scripts/preprocess_data.py
```

## 模型说明

### 基线双塔模型（Two-Tower）

传统的双塔模型在两个独立的塔（用户塔和物品塔）中处理特征，并在最后通过点积计算相似度。这种结构在推荐系统中被广泛使用，具有高效的推理性能，但表达能力有限。

### IntTower模型

IntTower通过引入三个核心组件增强了双塔模型：

1. **Light-SE**：轻量级的Squeeze-and-Excitation注意力机制，通过自适应加权不同的特征嵌入，强调重要特征。与标准SENET相比，Light-SE使用Softmax激活，确保特征权重之和为1。

2. **FE-Block**：细粒度早期特征交互模块，让用户塔各层的表示与物品塔的最终表示进行多头交互。这种设计在保持双塔高效特性的同时，增强了模型的表达能力。

3. **CIR**：对比交互正则化，通过对比学习增强表示空间的结构，使相关的用户-物品对靠近，不相关的远离。

## 运行实验
注：内存不足请在命令行使用--num_workers 0
### 一键运行所有实验

```bash
python run_all_experiments.py
```

### 分步运行

#### 基线模型
```bash
python scripts/run_two_tower.py
```

#### IntTower完整模型
```bash
python scripts/run_inttower.py --use_light_se --use_fe_block --use_cir
```

#### 消融实验
```bash
# 无Light-SE
python scripts/run_inttower.py --use_fe_block --use_cir

# 无FE-Block
python scripts/run_inttower.py --use_light_se --use_cir

# 无CIR
python scripts/run_inttower.py --use_light_se --use_fe_block

# 使用SENET代替Light-SE
python scripts/run_inttower.py --use_senet --use_fe_block --use_cir

# 使用FC层代替FE-Block
python scripts/run_inttower.py --use_light_se --use_fc --use_cir
```

#### 可视化结果
```bash
python scripts/visualize_results.py
```

## 参数配置

所有的训练脚本都支持通过命令行参数进行配置。主要参数包括：

- `--data_dir`: 指定数据目录
- `--batch_size`: 批次大小
- `--embedding_dim`: 特征嵌入维度
- `--mlp_dims`: MLP各层维度，如"256,128,128"
- `--dropout`: Dropout比例
- `--lr`: 学习率
- `--epochs`: 训练轮数
- `--gpu`: GPU编号，-1表示使用CPU

IntTower特有参数：
- `--use_light_se`: 使用Light-SE
- `--use_senet`: 使用标准SENET
- `--use_fe_block`: 使用FE-Block
- `--use_cir`: 使用CIR
- `--use_fc`: 使用FC层替代FE-Block
- `--cir_weight`: CIR损失的权重

完整参数列表请查看各脚本的帮助信息（`python scripts/run_inttower.py --help`）。

## 结果

实验结果将保存在`results`目录中：
- `results/two_tower/`: 基线模型结果
- `results/inttower/`: IntTower模型及消融实验结果
- `results/figures/`: 结果可视化图表



## 参考

[1] 《IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System》 