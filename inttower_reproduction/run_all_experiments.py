"""
运行完整的实验流水线:
1. 预处理数据
2. 训练基线Two-Tower模型
3. 训练完整的IntTower模型
4. 运行各种消融实验
5. 可视化结果
"""

import os
import sys
import argparse
import subprocess
import time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行IntTower所有实验")
    
    parser.add_argument("--skip_preprocess", action="store_true", default=False, 
                        help="跳过数据预处理")
    parser.add_argument("--skip_two_tower", action="store_true", default=False, 
                        help="跳过Two-Tower训练")
    parser.add_argument("--skip_inttower", action="store_true", default=False, 
                        help="跳过IntTower训练")
    parser.add_argument("--skip_ablation", action="store_true", default=False, 
                        help="跳过消融实验")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU编号，-1表示使用CPU")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="数据加载的并行工作进程数，传递给训练脚本")
    
    return parser.parse_args()


def run_command(cmd):
    """运行shell命令并打印输出"""
    print(f"执行命令: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True
    )
    
    if process.stdout is not None:
        for line in process.stdout:
            sys.stdout.write(line)
    
    process.wait()
    if process.returncode != 0:
        print(f"命令执行失败，退出代码: {process.returncode}")
        return False
    
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 预处理数据
    if not args.skip_preprocess:
        print("\n=== 步骤1: 预处理数据 ===\n")
        success = run_command("python scripts/preprocess_data.py")
        if not success:
            print("数据预处理失败，退出")
            return
    
    # 2. 训练Two-Tower模型
    if not args.skip_two_tower:
        print("\n=== 步骤2: 训练基线Two-Tower模型 ===\n")
        cmd = f"python scripts/run_two_tower.py --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        success = run_command(cmd)
        if not success:
            print("Two-Tower训练失败，但继续执行")
    
    # 3. 训练完整的IntTower模型
    if not args.skip_inttower:
        print("\n=== 步骤3: 训练完整IntTower模型 ===\n")
        cmd = f"python scripts/run_inttower.py --use_light_se --use_fe_block --use_cir --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        success = run_command(cmd)
        if not success:
            print("IntTower训练失败，但继续执行")
    
    # 4. 运行消融实验
    if not args.skip_ablation:
        print("\n=== 步骤4: 运行消融实验 ===\n")
        
        # a. 无Light-SE
        print("\n--- 消融实验: 无Light-SE ---\n")
        cmd = f"python scripts/run_inttower.py --use_fe_block --use_cir --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        run_command(cmd)
        
        # b. 无FE-Block
        print("\n--- 消融实验: 无FE-Block ---\n")
        cmd = f"python scripts/run_inttower.py --use_light_se --use_cir --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        run_command(cmd)
        
        # c. 无CIR
        print("\n--- 消融实验: 无CIR ---\n")
        cmd = f"python scripts/run_inttower.py --use_light_se --use_fe_block --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        run_command(cmd)
        
        # d. 使用SENET
        print("\n--- 消融实验: 使用SENET ---\n")
        cmd = f"python scripts/run_inttower.py --use_senet --use_fe_block --use_cir --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        run_command(cmd)
        
        # e. 使用FC
        print("\n--- 消融实验: 使用FC ---\n")
        cmd = f"python scripts/run_inttower.py --use_light_se --use_fc --use_cir --gpu {args.gpu}"
        if args.num_workers is not None:
            cmd += f" --num_workers {args.num_workers}"
        run_command(cmd)
    
    # 5. 可视化结果
    print("\n=== 步骤5: 可视化结果 ===\n")
    run_command("python scripts/visualize_results.py")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n=== 所有实验完成! 总运行时间: {hours}小时 {minutes}分钟 {seconds}秒 ===")


if __name__ == "__main__":
    main() 