"""
MovieLens-1M数据预处理脚本
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import run_preprocessing


if __name__ == "__main__":
    print("开始MovieLens-1M数据预处理...")
    run_preprocessing()
    print("数据预处理完成!") 