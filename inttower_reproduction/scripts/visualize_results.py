"""
可视化实验结果，用于比较Two-Tower和IntTower以及消融实验
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import visualize_ablation_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="可视化实验结果")
    
    parser.add_argument("--results_dir", type=str, default="./results", 
                        help="结果目录")
    parser.add_argument("--output_dir", type=str, default="./results/figures", 
                        help="图表输出目录")
    
    return parser.parse_args()


def load_results(results_dir):
    """
    加载所有实验结果
    
    参数:
        results_dir: 结果目录
        
    返回:
        all_results: 包含所有模型结果的字典
    """
    all_results = {}
    
    # 加载Two-Tower结果
    two_tower_result_path = os.path.join(results_dir, "two_tower", "results.json")
    if os.path.exists(two_tower_result_path):
        with open(two_tower_result_path, "r") as f:
            two_tower_results = json.load(f)
            all_results["Two-Tower"] = {
                "auc": two_tower_results["best_auc"],
                "logloss": two_tower_results["best_logloss"]
            }
    
    # 加载IntTower全部结果
    inttower_results_path = os.path.join(results_dir, "inttower", "all_results.json")
    if os.path.exists(inttower_results_path):
        with open(inttower_results_path, "r") as f:
            inttower_results = json.load(f)
            
            # 重命名实验
            name_mapping = {
                "inttower_full": "IntTower",
                "wo_light_se": "w/o Light-SE",
                "wo_fe_block": "w/o FE-Block",
                "wo_cir": "w/o CIR",
                "w_senet": "w/ SENET",
                "w_fc": "w/ FC"
            }
            
            for exp_name, results in inttower_results.items():
                display_name = name_mapping.get(exp_name, exp_name)
                all_results[display_name] = results
    
    return all_results


def visualize_table2(all_results, output_dir):
    """
    可视化Table 2 (Two-Tower vs. IntTower)的结果
    
    参数:
        all_results: 包含所有模型结果的字典
        output_dir: 输出目录
    """
    if "Two-Tower" not in all_results or "IntTower" not in all_results:
        print("无法创建Table 2比较图，缺少Two-Tower或IntTower的结果")
        return
    
    table2_results = {
        "Two-Tower": all_results["Two-Tower"],
        "IntTower": all_results["IntTower"]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "table2_comparison.png")
    
    visualize_ablation_results(table2_results, output_path)
    
    # 创建表格
    models = ["Two-Tower", "IntTower"]
    aucs = [all_results[model]["auc"] for model in models]
    loglosses = [all_results[model]["logloss"] for model in models]
    
    table_data = pd.DataFrame({
        "Model": models,
        "AUC": [f"{auc:.4f}" for auc in aucs],
        "Logloss": [f"{logloss:.4f}" for logloss in loglosses]
    })
    
    print("\nTable 2 复现结果:")
    print(table_data.to_string(index=False))
    
    # 输出到CSV文件
    table_data.to_csv(os.path.join(output_dir, "table2_comparison.csv"), index=False)


def visualize_figure6a(all_results, output_dir):
    """
    可视化Figure 6(a) (IntTower消融研究)的结果
    
    参数:
        all_results: 包含所有模型结果的字典
        output_dir: 输出目录
    """
    # 排序顺序
    expected_models = ["IntTower", "w/o Light-SE", "w/o FE-Block", "w/o CIR", "w/ SENET", "w/ FC"]
    
    # 筛选存在的模型
    ablation_results = {}
    for model in expected_models:
        if model in all_results:
            ablation_results[model] = all_results[model]
    
    if len(ablation_results) < 2:
        print("无法创建Figure 6(a)消融研究图，至少需要两个消融实验的结果")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure6a_ablation.png")
    
    visualize_ablation_results(ablation_results, output_path)
    
    # 创建表格
    models = list(ablation_results.keys())
    aucs = [ablation_results[model]["auc"] for model in models]
    loglosses = [ablation_results[model]["logloss"] for model in models]
    
    table_data = pd.DataFrame({
        "Model": models,
        "AUC": [f"{auc:.4f}" for auc in aucs],
        "Logloss": [f"{logloss:.4f}" for logloss in loglosses]
    })
    
    print("\nFigure 6(a) 消融实验结果:")
    print(table_data.to_string(index=False))
    
    # 输出到CSV文件
    table_data.to_csv(os.path.join(output_dir, "figure6a_ablation.csv"), index=False)


def create_metrics_comparison(results_dir, output_dir):
    """
    创建训练指标的比较图
    
    参数:
        results_dir: 结果目录
        output_dir: 输出目录
    """
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Two-Tower指标
    two_tower_metrics_path = os.path.join(results_dir, "two_tower", "metrics.pkl")
    
    # 加载IntTower指标
    inttower_metrics_path = os.path.join(results_dir, "inttower", "inttower_full", "metrics.pkl")
    
    if not os.path.exists(two_tower_metrics_path) or not os.path.exists(inttower_metrics_path):
        print("无法创建指标比较图，缺少Two-Tower或IntTower的指标数据")
        return
    
    with open(two_tower_metrics_path, "rb") as f:
        two_tower_metrics = pickle.load(f)
    
    with open(inttower_metrics_path, "rb") as f:
        inttower_metrics = pickle.load(f)
    
    # 创建AUC对比图
    plt.figure(figsize=(12, 6))
    plt.plot(two_tower_metrics["epochs"], two_tower_metrics["test_aucs"], label="Two-Tower")
    plt.plot(inttower_metrics["epochs"], inttower_metrics["test_aucs"], label="IntTower")
    plt.title("Test AUC Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "auc_comparison.png"))
    plt.close()
    
    # 创建Logloss对比图
    plt.figure(figsize=(12, 6))
    plt.plot(two_tower_metrics["epochs"], two_tower_metrics["test_loglosses"], label="Two-Tower")
    plt.plot(inttower_metrics["epochs"], inttower_metrics["test_loglosses"], label="IntTower")
    plt.title("Test Logloss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Logloss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "logloss_comparison.png"))
    plt.close()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载所有实验结果
    all_results = load_results(args.results_dir)
    
    if not all_results:
        print("未找到任何实验结果。请先运行训练脚本。")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 可视化Table 2 (Two-Tower vs. IntTower)
    visualize_table2(all_results, args.output_dir)
    
    # 可视化Figure 6(a) (IntTower消融研究)
    visualize_figure6a(all_results, args.output_dir)
    
    # 创建训练指标比较
    create_metrics_comparison(args.results_dir, args.output_dir)
    
    print(f"所有图表已保存至: {args.output_dir}")


if __name__ == "__main__":
    main() 