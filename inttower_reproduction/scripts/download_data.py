"""
下载MovieLens-1M数据集并解压到指定目录
"""

import os
import sys
import argparse
import zipfile
import shutil
import requests
from tqdm import tqdm

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="下载MovieLens-1M数据集")
    
    parser.add_argument("--output_dir", type=str, default="./data/ml-1m", 
                        help="数据输出目录")
    
    return parser.parse_args()


def download_file(url, output_path, chunk_size=8192):
    """
    从指定URL下载文件到输出路径
    
    参数:
        url: 下载链接
        output_path: 输出文件路径
        chunk_size: 分块大小
        
    返回:
        bool: 下载是否成功
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        file_size = int(response.headers.get('content-length', 0))
        
        # 显示下载进度条
        progress_bar = tqdm(
            total=file_size, 
            unit='B', 
            unit_scale=True, 
            desc=os.path.basename(output_path)
        )
        
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
                    
        progress_bar.close()
        return True
    
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def extract_zip(zip_path, extract_path):
    """
    解压ZIP文件
    
    参数:
        zip_path: ZIP文件路径
        extract_path: 解压目标目录
        
    返回:
        bool: 解压是否成功
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取ZIP文件中的所有文件
            total_files = len(zip_ref.filelist)
            
            # 显示解压进度条
            progress_bar = tqdm(
                total=total_files, 
                unit='files', 
                desc="解压文件"
            )
            
            # 解压每个文件
            for file_info in zip_ref.filelist:
                zip_ref.extract(file_info, extract_path)
                progress_bar.update(1)
                
            progress_bar.close()
        return True
    
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()
    
    # MovieLens-1M数据集的URL
    ml_1m_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    
    # 创建临时目录和输出目录
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 下载ZIP文件
    zip_path = os.path.join(tmp_dir, "ml-1m.zip")
    print(f"下载MovieLens-1M数据集到 {zip_path}...")
    
    if download_file(ml_1m_url, zip_path):
        print("下载成功!")
        
        # 解压ZIP文件
        print(f"解压数据到 {tmp_dir}...")
        if extract_zip(zip_path, tmp_dir):
            print("解压成功!")
            
            # 移动文件到输出目录
            extracted_dir = os.path.join(tmp_dir, "ml-1m")
            if os.path.exists(extracted_dir):
                print(f"复制文件到 {args.output_dir}...")
                
                # 复制数据文件
                for filename in ["ratings.dat", "users.dat", "movies.dat", "README"]:
                    src = os.path.join(extracted_dir, filename)
                    dst = os.path.join(args.output_dir, filename)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        print(f"已复制: {filename}")
                
                print("数据集准备完成!")
            else:
                print(f"错误: 解压后的目录 {extracted_dir} 不存在")
        
        # 清理临时文件
        try:
            shutil.rmtree(tmp_dir)
            print(f"已删除临时目录 {tmp_dir}")
        except Exception as e:
            print(f"无法删除临时目录: {e}")
    else:
        print("下载失败，请检查网络连接或手动下载数据集")


if __name__ == "__main__":
    main() 