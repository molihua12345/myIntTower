# IntTower Reproduction Project

This project reproduces the methods and experimental results from the paper "IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System".

## Project Introduction

IntTower is an improved two-tower model designed to address the limited expressiveness of traditional two-tower models. It enhances the two-tower architecture through three core innovative components:

1. Light-SE: Lightweight Squeeze-and-Excitation attention mechanism
2. FE-Block: Fine-grained Early Feature Interaction module
3. CIR: Contrastive Interaction Regularization

## Project Structure

```
inttower_reproduction/
├── data/                        # Data directory
│   └── ml-1m/                   # Original MovieLens-1M dataset
│       └── processed/           # Preprocessed data
├── src/                         # Source code directory
│   ├── models/                  # Model implementations
│   │   ├── __init__.py          # Model package initialization
│   │   ├── two_tower.py         # Baseline two-tower model
│   │   └── inttower.py          # IntTower model and its components
│   ├── preprocessing.py         # Data preprocessing
│   ├── data_loader.py           # Data loading classes
│   └── utils.py                 # Utility functions
├── scripts/                     # Scripts directory
│   ├── download_data.py         # Data download script
│   ├── preprocess_data.py       # Data preprocessing script
│   ├── run_two_tower.py         # Baseline model training script
│   ├── run_inttower.py          # IntTower model training script
│   └── visualize_results.py     # Results visualization script
├── results/                     # Experiment results directory
├── run_all_experiments.py       # Main script to run all experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Environment Setup

```bash
# Create virtual environment
conda create -n inttower python=3.8
conda activate inttower

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

1. Download MovieLens-1M dataset:
```bash
python scripts/download_data.py
```

2. Preprocess data:
```bash
python scripts/preprocess_data.py
```

## Model Description

### Baseline Two-Tower Model

Traditional two-tower models process features in two separate towers (user tower and item tower) and calculate similarity using dot product at the end. This structure is widely used in recommender systems due to its efficient inference performance, but has limited expressiveness.

### IntTower Model

IntTower enhances the two-tower model by introducing three core components:

1. **Light-SE**: Lightweight Squeeze-and-Excitation attention mechanism that adaptively weights different feature embeddings, emphasizing important features. Compared to standard SENET, Light-SE uses Softmax activation to ensure feature weights sum to 1.

2. **FE-Block**: Fine-grained Early Feature Interaction module that enables multi-head interaction between representations from each layer of the user tower and the final representation of the item tower. This design enhances the model's expressiveness while maintaining the efficient characteristics of two-tower models.

3. **CIR**: Contrastive Interaction Regularization, which enhances the structure of the representation space through contrastive learning, bringing relevant user-item pairs closer and pushing irrelevant ones apart.

## Running Experiments

### Run All Experiments in One Command

```bash
python run_all_experiments.py
```

### Run Step-by-Step

#### Baseline Model
```bash
python scripts/run_two_tower.py
```

#### Complete IntTower Model
```bash
python scripts/run_inttower.py --use_light_se --use_fe_block --use_cir
```

#### Ablation Experiments
```bash
# Without Light-SE
python scripts/run_inttower.py --use_fe_block --use_cir

# Without FE-Block
python scripts/run_inttower.py --use_light_se --use_cir

# Without CIR
python scripts/run_inttower.py --use_light_se --use_fe_block

# Using SENET instead of Light-SE
python scripts/run_inttower.py --use_senet --use_fe_block --use_cir

# Using FC layer instead of FE-Block
python scripts/run_inttower.py --use_light_se --use_fc --use_cir
```

#### Visualize Results
```bash
python scripts/visualize_results.py
```

## Parameter Configuration

All training scripts support configuration via command-line parameters. The main parameters include:

- `--data_dir`: Specify data directory
- `--batch_size`: Batch size
- `--embedding_dim`: Feature embedding dimension
- `--mlp_dims`: Dimensions of MLP layers, e.g., "256,128,128"
- `--dropout`: Dropout ratio
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--gpu`: GPU number, -1 means using CPU

IntTower specific parameters:
- `--use_light_se`: Use Light-SE
- `--use_senet`: Use standard SENET
- `--use_fe_block`: Use FE-Block
- `--use_cir`: Use CIR
- `--use_fc`: Use FC layer instead of FE-Block
- `--cir_weight`: Weight of CIR loss

For a complete list of parameters, check the help information of each script (e.g., `python scripts/run_inttower.py --help`).

## Results

Experimental results will be saved in the `results` directory:
- `results/two_tower/`: Baseline model results
- `results/inttower/`: IntTower model and ablation experiment results
- `results/figures/`: Result visualization charts

## References

[1] "IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System" 