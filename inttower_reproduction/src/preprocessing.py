import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MovieLensDataProcessor:
    """处理MovieLens-1M数据集的类"""
    
    def __init__(self, data_dir="./data/ml-1m"):
        """
        初始化数据处理器
        
        参数:
            data_dir: MovieLens-1M数据所在的目录
        """
        self.data_dir = data_dir
        self.user_data = None
        self.movie_data = None
        self.rating_data = None
        self.merged_data = None
        self.train_data = None
        self.test_data = None
        self.user_features = None
        self.item_features = None
        self.user_feature_dims = {}
        self.item_feature_dims = {}
        self.genre_list = None
        
    def load_data(self):
        """加载原始数据文件"""
        print("加载数据...")
        
        # 加载用户数据
        self.user_data = pd.read_csv(
            os.path.join(self.data_dir, "users.dat"),
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
            encoding="ISO-8859-1"
        )
        
        # 加载电影数据
        self.movie_data = pd.read_csv(
            os.path.join(self.data_dir, "movies.dat"),
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            encoding="ISO-8859-1"
        )
        
        # 加载评分数据
        self.rating_data = pd.read_csv(
            os.path.join(self.data_dir, "ratings.dat"),
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            encoding="ISO-8859-1"
        )
        
        print(f"用户数量: {len(self.user_data)}")
        print(f"电影数量: {len(self.movie_data)}")
        print(f"评分数量: {len(self.rating_data)}")
        
        return self
    
    def merge_data(self):
        """合并用户、电影和评分数据"""
        print("合并数据...")
        
        # 合并评分数据与用户数据
        merged_rating_user = pd.merge(
            self.rating_data, self.user_data, on="UserID", how="left"
        )
        
        # 将结果与电影数据合并
        self.merged_data = pd.merge(
            merged_rating_user, self.movie_data, on="MovieID", how="left"
        )
        
        print(f"合并后的数据条数: {len(self.merged_data)}")
        
        return self
    
    def generate_label(self, threshold=4):
        """根据评分生成二元标签"""
        print(f"根据阈值{threshold}生成标签...")
        
        self.merged_data["Label"] = (self.merged_data["Rating"] >= threshold).astype(int)
        
        # 统计正负样本比例
        positive_samples = self.merged_data["Label"].sum()
        negative_samples = len(self.merged_data) - positive_samples
        
        print(f"正样本数量: {positive_samples} ({positive_samples / len(self.merged_data):.2%})")
        print(f"负样本数量: {negative_samples} ({negative_samples / len(self.merged_data):.2%})")
        
        return self
    
    def process_features(self):
        """处理类别特征，进行编码"""
        print("处理特征...")
        
        # 用户特征处理
        # 对UserID进行编码
        unique_user_ids = self.merged_data["UserID"].unique()
        user_id_map = {id: idx for idx, id in enumerate(unique_user_ids)}
        self.merged_data["UserID_Idx"] = self.merged_data["UserID"].map(user_id_map)
        self.user_feature_dims["UserID"] = len(user_id_map)
        
        # 对Gender进行编码
        gender_map = {"M": 0, "F": 1}
        self.merged_data["Gender_Idx"] = self.merged_data["Gender"].map(gender_map)
        self.user_feature_dims["Gender"] = len(gender_map)
        
        # 对Age进行编码 (已分箱)
        age_map = {
            1: 0,  # "Under 18"
            18: 1,  # "18-24"
            25: 2,  # "25-34"
            35: 3,  # "35-44"
            45: 4,  # "45-49"
            50: 5,  # "50-55"
            56: 6,  # "56+"
        }
        self.merged_data["Age_Idx"] = self.merged_data["Age"].map(age_map)
        self.user_feature_dims["Age"] = len(age_map)
        
        # 对Occupation进行编码
        occupation_values = self.merged_data["Occupation"].unique()
        occupation_map = {val: idx for idx, val in enumerate(occupation_values)}
        self.merged_data["Occupation_Idx"] = self.merged_data["Occupation"].map(occupation_map)
        self.user_feature_dims["Occupation"] = len(occupation_map)
        
        # 电影特征处理
        # 对MovieID进行编码
        unique_movie_ids = self.merged_data["MovieID"].unique()
        movie_id_map = {id: idx for idx, id in enumerate(unique_movie_ids)}
        self.merged_data["MovieID_Idx"] = self.merged_data["MovieID"].map(movie_id_map)
        self.item_feature_dims["MovieID"] = len(movie_id_map)
        
        # 处理Genres特征（多值类别）
        # 提取所有类型列表
        all_genres = set()
        for genres in self.merged_data["Genres"].unique():
            all_genres.update(genres.split("|"))
        
        self.genre_list = sorted(list(all_genres))
        print(f"电影类型总数: {len(self.genre_list)}")
        
        # 创建类型到索引的映射
        self.genre_map = {genre: idx for idx, genre in enumerate(self.genre_list)}
        self.item_feature_dims["Genres"] = len(self.genre_list)
        
        # 将电影类型转换为多热编码的索引列表
        self.merged_data["Genres_Idx"] = self.merged_data["Genres"].apply(
            lambda x: [self.genre_map[genre] for genre in x.split("|")]
        )
        
        print("用户特征维度:", self.user_feature_dims)
        print("物品特征维度:", self.item_feature_dims)
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print(f"划分数据集 (测试集比例: {test_size})...")
        
        # 使用分层抽样，确保每个用户的样本都按比例分配到训练集和测试集
        # 这里使用用户ID作为分层抽样的依据
        train_indices, test_indices = train_test_split(
            np.arange(len(self.merged_data)),
            test_size=test_size,
            random_state=random_state,
            stratify=self.merged_data["UserID"]
        )
        
        self.train_data = self.merged_data.iloc[train_indices].reset_index(drop=True)
        self.test_data = self.merged_data.iloc[test_indices].reset_index(drop=True)
        
        print(f"训练集大小: {len(self.train_data)}")
        print(f"测试集大小: {len(self.test_data)}")
        
        return self
    
    def save_processed_data(self, output_dir="./data/ml-1m/processed"):
        """保存处理后的数据"""
        print("保存处理后的数据...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集和测试集
        self.train_data.to_pickle(os.path.join(output_dir, "train_data.pkl"))
        self.test_data.to_pickle(os.path.join(output_dir, "test_data.pkl"))
        
        # 保存特征维度信息
        feature_dims = {
            "user_feature_dims": self.user_feature_dims,
            "item_feature_dims": self.item_feature_dims,
            "genre_list": self.genre_list
        }
        
        import pickle
        with open(os.path.join(output_dir, "feature_info.pkl"), "wb") as f:
            pickle.dump(feature_dims, f)
        
        print(f"数据保存完成: {output_dir}")
        
        return self

def run_preprocessing():
    """执行数据预处理的主函数"""
    processor = MovieLensDataProcessor()
    processor.load_data().merge_data().generate_label().process_features().split_data().save_processed_data()
    
if __name__ == "__main__":
    run_preprocessing() 