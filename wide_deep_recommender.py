"""
Wide & Deep 电影推荐系统
包含：数据处理、向量化、模型构建、召回、重排序
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, ndcg_score
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class DataProcessor:
    """数据处理器：负责读取和预处理 MovieLens 数据"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.genre_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载 MovieLens 数据"""
        print("正在加载数据...")
        
        # 加载评分数据
        ratings = pd.read_csv(
            f'{self.data_path}/ratings.dat',
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        # 加载电影数据
        movies = pd.read_csv(
            f'{self.data_path}/movies.dat',
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )
        
        # 加载标签数据（如果存在）
        try:
            tags = pd.read_csv(
                f'{self.data_path}/tags.dat',
                sep='::',
                engine='python',
                names=['user_id', 'movie_id', 'tag', 'timestamp'],
                encoding='latin-1'
            )
        except:
            tags = None
            
        print(f"评分数据: {ratings.shape}")
        print(f"电影数据: {movies.shape}")
        
        return ratings, movies, tags
    
    def process_genres(self, movies):
        """处理电影类型数据"""
        print("处理电影类型...")
        
        # 获取所有类型
        all_genres = set()
        for genres_str in movies['genres']:
            genres = genres_str.split('|')
            all_genres.update(genres)
        
        all_genres = sorted(list(all_genres))
        print(f"找到 {len(all_genres)} 种电影类型: {all_genres}")
        
        # 为每个电影创建类型的 multi-hot 编码
        genre_matrix = np.zeros((len(movies), len(all_genres)))
        
        for idx, genres_str in enumerate(movies['genres']):
            genres = genres_str.split('|')
            for genre in genres:
                if genre in all_genres:
                    genre_idx = all_genres.index(genre)
                    genre_matrix[idx, genre_idx] = 1
        
        # 创建类型特征列
        for idx, genre in enumerate(all_genres):
            movies[f'genre_{genre}'] = genre_matrix[:, idx]
        
        return movies, all_genres
    
    def create_user_features(self, ratings, movies):
        """创建用户特征（包含 TF-IDF 类型偏好）"""
        print("创建用户特征...")
        
        # 用户统计特征
        user_stats = ratings.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_stats.columns = ['user_id', 'avg_rating', 'std_rating', 
                              'rating_count', 'first_rating_time', 'last_rating_time']
        
        # 填充缺失值
        user_stats['std_rating'].fillna(0, inplace=True)
        
        # 用户活跃度（评分时间跨度）
        user_stats['activity_days'] = (
            user_stats['last_rating_time'] - user_stats['first_rating_time']
        ) / (24 * 3600)
        
        # 计算用户类型偏好的 TF-IDF
        print("计算 TF-IDF 类型偏好...")
        user_genre_tfidf = self._compute_user_genre_tfidf(ratings, movies)
        user_stats = user_stats.merge(user_genre_tfidf, on='user_id', how='left')
        
        return user_stats
    
    def _compute_user_genre_tfidf(self, ratings, movies):
        """计算用户类型偏好的 TF-IDF 权重"""
        # 合并评分和电影类型
        user_movies = ratings.merge(movies[['movie_id', 'genres']], on='movie_id')
        
        # 只考虑高评分（>=4）的电影
        user_movies = user_movies[user_movies['rating'] >= 4]
        
        # 计算 TF：用户对每个类型的评分频率
        user_genre_tf = defaultdict(lambda: defaultdict(int))
        for _, row in user_movies.iterrows():
            user_id = row['user_id']
            for genre in row['genres'].split('|'):
                user_genre_tf[user_id][genre] += 1
        
        # 计算 IDF：类型的稀有程度
        total_users = ratings['user_id'].nunique()
        genre_user_count = defaultdict(int)
        for user_genres in user_genre_tf.values():
            for genre in user_genres.keys():
                genre_user_count[genre] += 1
        
        genre_idf = {}
        for genre, count in genre_user_count.items():
            genre_idf[genre] = np.log(total_users / (count + 1))
        
        # 计算 TF-IDF 并构造特征
        user_tfidf_features = []
        for user_id in ratings['user_id'].unique():
            user_tf = user_genre_tf.get(user_id, {})
            tfidf_dict = {'user_id': user_id}
            
            for genre, tf in user_tf.items():
                idf = genre_idf.get(genre, 0)
                tfidf_dict[f'tfidf_{genre}'] = tf * idf
            
            user_tfidf_features.append(tfidf_dict)
        
        return pd.DataFrame(user_tfidf_features).fillna(0)
    
    def create_movie_features(self, ratings, movies):
        """创建电影特征"""
        print("创建电影特征...")
        
        # 电影统计特征
        movie_stats = ratings.groupby('movie_id').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        movie_stats.columns = ['movie_id', 'avg_rating', 'std_rating',
                               'rating_count', 'first_rating_time', 'last_rating_time']
        
        # 填充缺失值
        movie_stats['std_rating'].fillna(0, inplace=True)
        
        # 电影流行度
        movie_stats['popularity'] = np.log1p(movie_stats['rating_count'])
        
        # 合并电影信息
        movie_features = movies.merge(movie_stats, on='movie_id', how='left')
        
        return movie_features
    
    def prepare_training_data(self, ratings, user_stats, movie_features, 
                             test_size=0.2, negative_samples=1):
        """准备训练数据（包括负采样）"""
        print("准备训练数据...")
        
        # ⚡ 内存优化：只使用部分数据
        print("为节省内存，采样 50% 的评分数据...")
        ratings = ratings.sample(frac=0.5, random_state=42).reset_index(drop=True)
        
        # 合并用户和电影特征
        data = ratings.merge(user_stats, on='user_id', how='left')
        data = data.merge(movie_features, on='movie_id', how='left')
        
        # 创建标签：评分>=4为正样本，<3为负样本
        data['label'] = (data['rating'] >= 4).astype(int)
        
        # ⚡ 减少负采样比例：从 4 降到 1
        print(f"进行负采样（每个正样本对应 {negative_samples} 个负样本）...")
        all_movie_ids = set(movie_features['movie_id'].values)
        
        negative_samples_list = []
        user_movie_pairs = data.groupby('user_id')['movie_id'].apply(set).to_dict()
        
        # ⚡ 限制处理的用户数
        sampled_users = list(user_movie_pairs.keys())[:50000]  # 最多处理 5 万用户
        
        for user_id in sampled_users:
            rated_movies = user_movie_pairs[user_id]
            unrated_movies = list(all_movie_ids - rated_movies)
            
            # 每个用户最多采样固定数量的负样本
            num_neg = min(negative_samples * len(rated_movies), 100)  # 每用户最多 100 个负样本
            
            if len(unrated_movies) > num_neg:
                neg_samples = np.random.choice(unrated_movies, size=num_neg, replace=False)
            else:
                neg_samples = unrated_movies[:num_neg]
            
            for movie_id in neg_samples:
                negative_samples_list.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'label': 0
                })
        
        print(f"生成 {len(negative_samples_list)} 个负样本...")
        
        # 合并正负样本
        neg_data = pd.DataFrame(negative_samples_list)
        neg_data = neg_data.merge(user_stats, on='user_id', how='left')
        neg_data = neg_data.merge(movie_features, on='movie_id', how='left')
        
        # 清理内存
        del negative_samples_list
        
        # 只保留正样本的标签列
        data = pd.concat([data, neg_data], ignore_index=True)
        del neg_data
        
        # 打乱数据
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"训练数据总量: {len(data):,}")
        print(f"正样本: {data['label'].sum()}, 负样本: {len(data) - data['label'].sum()}")
        
        return data


class MovieDataset(Dataset):
    """PyTorch 数据集"""
    
    def __init__(self, user_ids, movie_ids, user_stats, movie_stats, genres, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.user_stats = torch.FloatTensor(user_stats)
        self.movie_stats = torch.FloatTensor(movie_stats)
        self.genres = torch.FloatTensor(genres)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.movie_ids[idx],
            self.user_stats[idx],
            self.movie_stats[idx],
            self.genres[idx],
            self.labels[idx]
        )


class WideDeepNet(nn.Module):
    """Wide & Deep 神经网络模型 (PyTorch) - 增强版"""
    
    def __init__(self, num_users, num_movies, num_genres,
                 embedding_dim=64, deep_layers=[512, 256, 128]):
        super(WideDeepNet, self).__init__()
        
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        
        # ========== Wide 部分：增强交叉特征 ==========
        # Wide 输入维度: 2(user+movie) + 4(user_stats) + 3(movie_stats) + num_genres
        # + 交叉特征: user*movie, user_avg*movie_avg, user_count*movie_popularity
        wide_input_dim = 2 + 4 + 3 + num_genres + 3  # 增加 3 个交叉特征
        self.wide_layer = nn.Linear(wide_input_dim, 1)
        
        # ========== Deep 部分：增大 Embedding 维度 ==========
        # Embedding 层：32 -> 64 维
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Deep 输入维度: 2*embedding_dim + 4(user_stats) + 3(movie_stats) + num_genres
        deep_input_dim = 2 * embedding_dim + 4 + 3 + num_genres
        
        # 构建深度网络层：[256,128,64] -> [512,256,128]
        deep_layers_list = []
        prev_dim = deep_input_dim
        
        for units in deep_layers:
            deep_layers_list.append(nn.Linear(prev_dim, units))
            deep_layers_list.append(nn.ReLU())
            deep_layers_list.append(nn.Dropout(0.3))
            deep_layers_list.append(nn.BatchNorm1d(units))
            prev_dim = units
        
        # Deep 输出层
        deep_layers_list.append(nn.Linear(prev_dim, 1))
        
        self.deep_layers = nn.Sequential(*deep_layers_list)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        nn.init.xavier_uniform_(self.wide_layer.weight)
        
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, movie_ids, user_stats, movie_stats, genres):
        """
        前向传播
        
        参数:
            user_ids: [batch_size]
            movie_ids: [batch_size]
            user_stats: [batch_size, 4] - [avg_rating, std_rating, rating_count, activity_days]
            movie_stats: [batch_size, 3] - [avg_rating, std_rating, popularity]
            genres: [batch_size, num_genres]
        """
        batch_size = user_ids.size(0)
        
        # ========== Wide 部分：添加交叉特征 ==========
        # 基础特征
        user_ids_float = user_ids.float().unsqueeze(1)  # [batch_size, 1]
        movie_ids_float = movie_ids.float().unsqueeze(1)  # [batch_size, 1]
        
        # 交叉特征
        user_movie_cross = (user_ids_float * movie_ids_float)  # user*movie ID 交叉
        user_avg_movie_avg_cross = (user_stats[:, 0:1] * movie_stats[:, 0:1])  # 平均评分交叉
        user_count_movie_pop_cross = (user_stats[:, 2:3] * movie_stats[:, 2:3])  # 评分数*流行度
        
        # 拼接所有特征作为 Wide 输入
        wide_input = torch.cat([
            user_ids_float,
            movie_ids_float,
            user_stats,  # [batch_size, 4]
            movie_stats,  # [batch_size, 3]
            genres,  # [batch_size, num_genres]
            user_movie_cross,  # [batch_size, 1]
            user_avg_movie_avg_cross,  # [batch_size, 1]
            user_count_movie_pop_cross  # [batch_size, 1]
        ], dim=1)
        
        wide_output = self.wide_layer(wide_input)  # [batch_size, 1]
        
        # ========== Deep 部分 ==========
        # Embedding
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]
        
        # 拼接所有特征作为 Deep 输入
        deep_input = torch.cat([
            user_emb,
            movie_emb,
            user_stats,
            movie_stats,
            genres
        ], dim=1)
        
        deep_output = self.deep_layers(deep_input)  # [batch_size, 1]
        
        # ========== 联合输出 ==========
        output = wide_output + deep_output  # [batch_size, 1]
        output = torch.sigmoid(output)  # [batch_size, 1]
        
        return output.squeeze()  # [batch_size]


class WideDeepModel:
    """Wide & Deep 推荐模型 (PyTorch 版本) - 增强版"""
    
    def __init__(self, num_users, num_movies, num_genres, 
                 embedding_dim=64, deep_layers=[512, 256, 128]):
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.embedding_dim = embedding_dim
        self.deep_layers = deep_layers
        self.model = None
        self.device = device
        
    def build_model(self):
        """构建 Wide & Deep 模型"""
        print("构建 Wide & Deep 模型 (PyTorch)...")
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = WideDeepNet(
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_genres=self.num_genres,
            embedding_dim=self.embedding_dim,
            deep_layers=self.deep_layers
        ).to(self.device)
        
        # 打印模型结构
        print("\n模型结构:")
        print(self.model)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return self.model
    
    def train(self, train_data, val_data, epochs=10, batch_size=1024, learning_rate=0.001):
        """训练模型"""
        print("\n开始训练 (PyTorch)...")
        print(f"设备: {self.device}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {learning_rate}")
        
        # 准备训练数据
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 创建数据集和数据加载器
        train_dataset = MovieDataset(
            X_train[0].flatten(), X_train[1].flatten(),
            X_train[2], X_train[3], X_train[4], y_train
        )
        val_dataset = MovieDataset(
            X_val[0].flatten(), X_val[1].flatten(),
            X_val[2], X_val[3], X_val[4], y_val
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.BCELoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=True
        )
        
        # 训练历史记录
        history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': []
        }
        
        best_val_auc = 0.0
        patience_counter = 0
        patience = 3
        
        # 训练循环
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # ========== 训练阶段 ==========
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []
            
            for batch_idx, (user_ids, movie_ids, user_stats, movie_stats, genres, labels) in enumerate(train_loader):
                # 移动到设备
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                user_stats = user_stats.to(self.device)
                movie_stats = movie_stats.to(self.device)
                genres = genres.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(user_ids, movie_ids, user_stats, movie_stats, genres)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item() * user_ids.size(0)
                predicted = (outputs >= 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
                
                # 打印进度
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # 计算训练指标
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            train_auc = roc_auc_score(train_labels, train_preds)
            
            # ========== 验证阶段 ==========
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for user_ids, movie_ids, user_stats, movie_stats, genres, labels in val_loader:
                    # 移动到设备
                    user_ids = user_ids.to(self.device)
                    movie_ids = movie_ids.to(self.device)
                    user_stats = user_stats.to(self.device)
                    movie_stats = movie_stats.to(self.device)
                    genres = genres.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(user_ids, movie_ids, user_stats, movie_stats, genres)
                    loss = criterion(outputs, labels)
                    
                    # 统计
                    val_loss += loss.item() * user_ids.size(0)
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # 计算验证指标
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            val_auc = roc_auc_score(val_labels, val_preds)
            
            # 记录历史
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # 打印结果
            print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            
            # 学习率调度
            scheduler.step(val_auc)
            
            # Early Stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    # 恢复最佳模型
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        # 创建类似 Keras 的 history 对象
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return History(history)
    
    def predict(self, X, batch_size=1024):
        """预测"""
        self.model.eval()
        
        # 准备数据
        user_ids = torch.LongTensor(X[0].flatten()).to(self.device)
        movie_ids = torch.LongTensor(X[1].flatten()).to(self.device)
        user_stats = torch.FloatTensor(X[2]).to(self.device)
        movie_stats = torch.FloatTensor(X[3]).to(self.device)
        genres = torch.FloatTensor(X[4]).to(self.device)
        
        predictions = []
        
        # 分批预测
        with torch.no_grad():
            for i in range(0, len(user_ids), batch_size):
                batch_user = user_ids[i:i+batch_size]
                batch_movie = movie_ids[i:i+batch_size]
                batch_user_stats = user_stats[i:i+batch_size]
                batch_movie_stats = movie_stats[i:i+batch_size]
                batch_genres = genres[i:i+batch_size]
                
                outputs = self.model(
                    batch_user, batch_movie,
                    batch_user_stats, batch_movie_stats, batch_genres
                )
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'num_genres': self.num_genres,
            'embedding_dim': self.embedding_dim,
            'deep_layers': self.deep_layers
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 重建模型
        self.num_users = checkpoint['num_users']
        self.num_movies = checkpoint['num_movies']
        self.num_genres = checkpoint['num_genres']
        self.embedding_dim = checkpoint['embedding_dim']
        self.deep_layers = checkpoint['deep_layers']
        
        self.model = WideDeepNet(
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_genres=self.num_genres,
            embedding_dim=self.embedding_dim,
            deep_layers=self.deep_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型已从 {path} 加载")


class RecallEngine:
    """召回引擎：基于多路召回策略"""
    
    def __init__(self, ratings, movies, movie_features):
        self.ratings = ratings
        self.movies = movies
        self.movie_features = movie_features
        
    def popular_recall(self, top_k=100):
        """热门召回：返回最受欢迎的电影"""
        popular_movies = self.movie_features.nlargest(top_k, 'rating_count')
        return popular_movies['movie_id'].tolist()
    
    def genre_based_recall(self, user_id, top_k=50):
        """基于类型的召回：根据用户喜好的类型推荐"""
        # 获取用户评分过的电影
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        user_movies = user_ratings.merge(self.movies, on='movie_id')
        
        # 统计用户喜欢的类型（评分>=4）
        liked_movies = user_movies[user_movies['rating'] >= 4]
        
        if len(liked_movies) == 0:
            return []
        
        # 提取类型
        genre_counts = {}
        for genres_str in liked_movies['genres']:
            for genre in genres_str.split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # 获取最喜欢的类型
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_genre_names = [g[0] for g in top_genres]
        
        # 找到包含这些类型的电影
        candidate_movies = self.movies[
            self.movies['genres'].str.contains('|'.join(top_genre_names))
        ]
        
        # 排除已评分的电影
        rated_movie_ids = set(user_ratings['movie_id'].values)
        candidate_movies = candidate_movies[
            ~candidate_movies['movie_id'].isin(rated_movie_ids)
        ]
        
        # 合并评分信息，按流行度排序
        candidate_movies = candidate_movies.merge(
            self.movie_features[['movie_id', 'popularity']], 
            on='movie_id'
        )
        candidate_movies = candidate_movies.nlargest(top_k, 'popularity')
        
        return candidate_movies['movie_id'].tolist()
    
    def collaborative_recall(self, user_id, top_k=50):
        """协同过滤召回：基于相似用户"""
        # 获取用户评分
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return []
        
        # 找到评分过相同电影的其他用户
        user_movies = set(user_ratings['movie_id'].values)
        similar_users = self.ratings[
            self.ratings['movie_id'].isin(user_movies)
        ]['user_id'].value_counts()
        
        # 排除自己，取前20个相似用户
        similar_users = similar_users[similar_users.index != user_id].head(20)
        
        if len(similar_users) == 0:
            return []
        
        # 获取相似用户喜欢的电影
        similar_user_ratings = self.ratings[
            (self.ratings['user_id'].isin(similar_users.index)) &
            (self.ratings['rating'] >= 4)
        ]
        
        # 排除当前用户已评分的电影
        candidate_movies = similar_user_ratings[
            ~similar_user_ratings['movie_id'].isin(user_movies)
        ]['movie_id'].value_counts().head(top_k)
        
        return candidate_movies.index.tolist()
    
    def diversity_recall(self, user_id, top_k=50):
        """多样性召回：推荐不同类型的电影"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            # 冷启动：返回各个类型的热门电影
            all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                         'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                         'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
            diverse_movies = []
            for genre in all_genres[:10]:  # 取前10个类型
                genre_movies = self.movies[
                    self.movies['genres'].str.contains(genre)
                ].merge(self.movie_features[['movie_id', 'popularity']], on='movie_id')
                if len(genre_movies) > 0:
                    top_movie = genre_movies.nlargest(1, 'popularity')['movie_id'].values
                    if len(top_movie) > 0:
                        diverse_movies.append(top_movie[0])
            return diverse_movies[:top_k]
        
        # 获取用户已评分的类型
        user_movies = user_ratings.merge(self.movies, on='movie_id')
        user_genres = set()
        for genres_str in user_movies['genres']:
            user_genres.update(genres_str.split('|'))
        
        # 找到用户未接触过的类型
        all_genres = set()
        for genres_str in self.movies['genres']:
            all_genres.update(genres_str.split('|'))
        
        unexplored_genres = all_genres - user_genres
        
        # 从未探索类型中推荐热门电影
        diverse_movies = []
        rated_movie_ids = set(user_ratings['movie_id'].values)
        
        for genre in list(unexplored_genres)[:5]:  # 取前5个未探索类型
            genre_movies = self.movies[
                self.movies['genres'].str.contains(genre)
            ].merge(self.movie_features[['movie_id', 'popularity']], on='movie_id')
            
            genre_movies = genre_movies[
                ~genre_movies['movie_id'].isin(rated_movie_ids)
            ]
            
            if len(genre_movies) > 0:
                top_movies = genre_movies.nlargest(10, 'popularity')['movie_id'].tolist()
                diverse_movies.extend(top_movies)
        
        return diverse_movies[:top_k]
    
    def multi_recall(self, user_id, top_k=200):
        """多路召回：整合多种召回策略（增强版）"""
        recalled_movies = set()
        
        # 热门召回
        popular = self.popular_recall(top_k=50)
        recalled_movies.update(popular)
        print(f"  - 热门召回: {len(popular)} 部")
        
        # 类型召回
        genre_based = self.genre_based_recall(user_id, top_k=80)
        recalled_movies.update(genre_based)
        print(f"  - 类型召回: {len(genre_based)} 部")
        
        # 协同过滤召回
        collaborative = self.collaborative_recall(user_id, top_k=80)
        recalled_movies.update(collaborative)
        print(f"  - 协同过滤召回: {len(collaborative)} 部")
        
        # 多样性召回（新增）
        diversity = self.diversity_recall(user_id, top_k=50)
        recalled_movies.update(diversity)
        print(f"  - 多样性召回: {len(diversity)} 部")
        
        # 限制总数
        recalled_movies = list(recalled_movies)[:top_k]
        
        print(f"总召回候选电影数: {len(recalled_movies)}")
        
        return recalled_movies


class RerankEngine:
    """重排序引擎：使用 Wide & Deep 模型进行精排（增强版）"""
    
    def __init__(self, model, user_encoder, movie_encoder, 
                 user_stats, movie_features, all_genres, ratings, movies):
        self.model = model
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.user_stats = user_stats
        self.movie_features = movie_features
        self.all_genres = all_genres
        self.ratings = ratings
        self.movies = movies
        
    def prepare_features(self, user_id, movie_ids):
        """准备特征用于模型预测"""
        num_samples = len(movie_ids)
        
        # 用户ID编码
        user_id_encoded = self.user_encoder.transform([user_id])[0]
        user_ids = np.array([user_id_encoded] * num_samples)
        
        # 电影ID编码
        movie_ids_encoded = self.movie_encoder.transform(movie_ids)
        
        # 用户统计特征
        user_stat = self.user_stats[self.user_stats['user_id'] == user_id].iloc[0]
        user_stats_array = np.array([
            user_stat['avg_rating'],
            user_stat['std_rating'],
            user_stat['rating_count'],
            user_stat['activity_days']
        ])
        user_stats_matrix = np.tile(user_stats_array, (num_samples, 1))
        
        # 电影统计特征和类型特征 - 修复顺序问题
        movie_stats_list = []
        genre_list = []
        genre_cols = [f'genre_{g}' for g in self.all_genres]
        
        for movie_id in movie_ids:
            movie_row = self.movie_features[self.movie_features['movie_id'] == movie_id]
            
            if not movie_row.empty:
                movie_row = movie_row.iloc[0]
                # 电影统计特征
                movie_stats_list.append([
                    movie_row['avg_rating'],
                    movie_row['std_rating'],
                    movie_row['popularity']
                ])
                # 类型特征
                genre_list.append([movie_row[col] for col in genre_cols])
            else:
                # 使用默认值
                movie_stats_list.append([0.0, 0.0, 0.0])
                genre_list.append([0.0] * len(self.all_genres))
        
        movie_stats_matrix = np.array(movie_stats_list)
        genre_matrix = np.array(genre_list)
        
        return [
            user_ids.reshape(-1, 1),
            movie_ids_encoded.reshape(-1, 1),
            user_stats_matrix,
            movie_stats_matrix,
            genre_matrix
        ]
    
    def apply_diversity_control(self, candidate_movies, scores, lambda_param=0.5):
        """应用 MMR 多样性控制"""
        # 获取电影类型信息
        movie_genres = {}
        for movie_id in candidate_movies:
            movie_row = self.movies[self.movies['movie_id'] == movie_id]
            if not movie_row.empty:
                movie_genres[movie_id] = set(movie_row.iloc[0]['genres'].split('|'))
            else:
                movie_genres[movie_id] = set()
        
        # MMR 算法
        selected = []
        selected_genres = set()
        remaining = list(zip(candidate_movies, scores))
        
        # 选择第一部电影（分数最高）
        if remaining:
            best_movie, best_score = max(remaining, key=lambda x: x[1])
            selected.append((best_movie, best_score))
            selected_genres.update(movie_genres.get(best_movie, set()))
            remaining.remove((best_movie, best_score))
        
        # 迭代选择剩余电影
        while remaining and len(selected) < len(candidate_movies):
            best_mmr_score = -float('inf')
            best_item = None
            
            for movie_id, score in remaining:
                # 计算与已选电影的类型重叠度
                genres = movie_genres.get(movie_id, set())
                overlap = len(genres & selected_genres)
                diversity_penalty = overlap / (len(genres) + 1e-6)
                
                # MMR 分数：平衡相关性和多样性
                mmr_score = lambda_param * score - (1 - lambda_param) * diversity_penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = (movie_id, score)
            
            if best_item:
                selected.append(best_item)
                selected_genres.update(movie_genres.get(best_item[0], set()))
                remaining.remove(best_item)
            else:
                break
        
        return zip(*selected) if selected else ([], [])
    
    def remove_user_history(self, user_id, candidate_movies):
        """移除用户历史评分过的电影"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        rated_movie_ids = set(user_ratings['movie_id'].values)
        
        filtered_movies = [m for m in candidate_movies if m not in rated_movie_ids]
        print(f"  - 移除历史: {len(candidate_movies)} -> {len(filtered_movies)}")
        return filtered_movies
    
    def balance_genre_distribution(self, candidate_movies, scores, max_per_genre=3):
        """平衡类型分布，避免某个类型占比过高"""
        genre_count = defaultdict(int)
        balanced_movies = []
        balanced_scores = []
        
        for movie_id, score in zip(candidate_movies, scores):
            movie_row = self.movies[self.movies['movie_id'] == movie_id]
            if movie_row.empty:
                balanced_movies.append(movie_id)
                balanced_scores.append(score)
                continue
            
            genres = movie_row.iloc[0]['genres'].split('|')
            primary_genre = genres[0]  # 使用第一个类型作为主类型
            
            if genre_count[primary_genre] < max_per_genre:
                balanced_movies.append(movie_id)
                balanced_scores.append(score)
                genre_count[primary_genre] += 1
        
        return balanced_movies, balanced_scores
    
    def rerank(self, user_id, candidate_movies, top_k=10, 
               enable_diversity=True, enable_dedup=True, enable_balance=False):
        """对候选电影进行重排序（增强版）"""
        print(f"重排序 {len(candidate_movies)} 部候选电影...")
        
        # 步骤 1：移除用户历史
        if enable_dedup:
            candidate_movies = self.remove_user_history(user_id, candidate_movies)
            if len(candidate_movies) == 0:
                print("所有候选电影都已被评分")
                return [], []
        
        # 步骤 2：准备特征
        X = self.prepare_features(user_id, candidate_movies)
        
        # 步骤 3：预测评分
        scores = self.model.predict(X).flatten()
        
        print(f"\n[评分统计] 最小: {scores.min():.4f}, 最大: {scores.max():.4f}, "
              f"平均: {scores.mean():.4f}, 标准差: {scores.std():.4f}")
        
        # 步骤 4：排序
        ranked_indices = np.argsort(scores)[::-1]
        ranked_movies = [candidate_movies[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices]
        
        # 步骤 5：应用多样性控制
        if enable_diversity and len(ranked_movies) > top_k:
            print("  - 应用 MMR 多样性控制...")
            ranked_movies, ranked_scores = self.apply_diversity_control(
                ranked_movies[:top_k*3], ranked_scores[:top_k*3]
            )
            ranked_movies = list(ranked_movies)[:top_k]
            ranked_scores = list(ranked_scores)[:top_k]
        else:
            ranked_movies = ranked_movies[:top_k]
            ranked_scores = ranked_scores[:top_k].tolist()
        
        # 步骤 6：平衡类型分布（可选）
        if enable_balance:
            print("  - 平衡类型分布...")
            ranked_movies, ranked_scores = self.balance_genre_distribution(
                ranked_movies, ranked_scores
            )
        
        return ranked_movies, ranked_scores


class MovieRecommender:
    """完整的电影推荐系统"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.processor = DataProcessor(data_path)
        self.model = None
        self.recall_engine = None
        self.rerank_engine = None
        
    def prepare_data(self):
        """准备数据"""
        # 加载数据
        ratings, movies, tags = self.processor.load_data()
        
        # 处理类型
        movies, all_genres = self.processor.process_genres(movies)
        
        # 创建特征
        user_stats = self.processor.create_user_features(ratings, movies)
        movie_features = self.processor.create_movie_features(ratings, movies)
        
        # 准备训练数据
        train_data = self.processor.prepare_training_data(
            ratings, user_stats, movie_features
        )
        
        return ratings, movies, train_data, user_stats, movie_features, all_genres
    
    def build_and_train(self, train_data, user_stats, movie_features, all_genres,
                       epochs=10, batch_size=1024):
        """构建和训练模型"""
        # 编码用户和电影ID
        self.processor.user_encoder.fit(train_data['user_id'])
        self.processor.movie_encoder.fit(train_data['movie_id'])
        
        # 准备特征
        genre_cols = [f'genre_{g}' for g in all_genres]
        
        X_user = self.processor.user_encoder.transform(train_data['user_id'])
        X_movie = self.processor.movie_encoder.transform(train_data['movie_id'])
        X_user_stats = train_data[
            ['avg_rating_x', 'std_rating_x', 'rating_count_x', 'activity_days']
        ].fillna(0).values
        X_movie_stats = train_data[
            ['avg_rating_y', 'std_rating_y', 'popularity']
        ].fillna(0).values
        X_genres = train_data[genre_cols].fillna(0).values
        y = train_data['label'].values
        
        # 划分训练集和验证集
        indices = np.arange(len(train_data))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train = [
            X_user[train_idx].reshape(-1, 1),
            X_movie[train_idx].reshape(-1, 1),
            X_user_stats[train_idx],
            X_movie_stats[train_idx],
            X_genres[train_idx]
        ]
        y_train = y[train_idx]
        
        X_val = [
            X_user[val_idx].reshape(-1, 1),
            X_movie[val_idx].reshape(-1, 1),
            X_user_stats[val_idx],
            X_movie_stats[val_idx],
            X_genres[val_idx]
        ]
        y_val = y[val_idx]
        
        # 构建模型
        num_users = len(self.processor.user_encoder.classes_)
        num_movies = len(self.processor.movie_encoder.classes_)
        num_genres = len(all_genres)
        
        self.model = WideDeepModel(num_users, num_movies, num_genres)
        self.model.build_model()
        
        # 训练
        history = self.model.train(
            (X_train, y_train),
            (X_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def setup_engines(self, ratings, movies, movie_features, user_stats, all_genres):
        """设置召回和重排序引擎"""
        # 召回引擎
        self.recall_engine = RecallEngine(ratings, movies, movie_features)
        
        # 重排序引擎（增强版）
        self.rerank_engine = RerankEngine(
            self.model,
            self.processor.user_encoder,
            self.processor.movie_encoder,
            user_stats,
            movie_features,
            all_genres,
            ratings,  # 新增
            movies    # 新增
        )
    
    def recommend(self, user_id, top_k=10):
        """为用户推荐电影"""
        print(f"\n为用户 {user_id} 生成推荐...")
        
        # 召回阶段
        candidate_movies = self.recall_engine.multi_recall(user_id, top_k=200)
        
        if len(candidate_movies) == 0:
            print("没有找到候选电影")
            return [], []
        
        # 重排序阶段
        recommended_movies, scores = self.rerank_engine.rerank(
            user_id, candidate_movies, top_k=top_k
        )
        
        return recommended_movies, scores
    
    def get_movie_info(self, movie_ids, movies):
        """获取电影信息"""
        movie_info = movies[movies['movie_id'].isin(movie_ids)]
        return movie_info[['movie_id', 'title', 'genres']]
    
    def save(self, model_path='wide_deep_model.pth', processor_path='processor.pkl'):
        """保存模型和处理器"""
        self.model.save_model(model_path)
        
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        
        print(f"处理器已保存至: {processor_path}")

    def load(self, model_path='wide_deep_model.pth', processor_path='processor.pkl'):
        """加载模型和处理器"""
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        # 构建占位模型，随后加载真实权重
        self.model = WideDeepModel(0, 0, 0)  # 参数会被加载的模型覆盖
        self.model.load_model(model_path)
        
        print("模型和处理器加载完成")
    
    def analyze_user_embedding(self, user_ids=None, top_dims=10):
        """分析用户Embedding维度与用户统计/类型偏好的关系，打印简要报告。
        参数:
            user_ids: 可选，待分析的用户原始ID列表；默认随机抽样 5000 个已出现的用户
            top_dims: 展示前多少个维度
        """
        if self.model is None or self.model.model is None:
            print("模型未构建或未加载！")
            return
        # 取出已训练的用户embedding权重
        user_emb_weight = self.model.model.user_embedding.weight.detach().cpu().numpy()
        num_users, emb_dim = user_emb_weight.shape
        top_dims = min(top_dims, emb_dim)

        # 准备待分析的用户索引
        if user_ids is None:
            # 使用编码器中的类作为已知用户集合
            all_user_raw = np.array(self.processor.user_encoder.classes_)
            # 随机抽样最多5000个
            rng = np.random.default_rng(42)
            if len(all_user_raw) > 5000:
                sampled_raw = rng.choice(all_user_raw, size=5000, replace=False)
            else:
                sampled_raw = all_user_raw
        else:
            sampled_raw = np.array(user_ids)

        # 将原始ID映射为编码索引，过滤不可映射的ID
        valid_mask = np.isin(sampled_raw, self.processor.user_encoder.classes_)
        sampled_raw = sampled_raw[valid_mask]
        sampled_idx = self.processor.user_encoder.transform(sampled_raw)

        # 取用户统计特征
        stats = self.processor.create_user_features(
            # 仅使用这些用户的评分子集来避免全量内存压力
            # 这里通过 data_path 重新读取评分
            pd.read_csv(f"{self.data_path}/ratings.dat", sep='::', engine='python',
                        names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1'),
            pd.read_csv(f"{self.data_path}/movies.dat", sep='::', engine='python',
                        names=['movie_id', 'title', 'genres'], encoding='latin-1')
        )
        stats = stats[stats['user_id'].isin(sampled_raw)].set_index('user_id')
        # 用户类型偏好（每类占比）
        # 构造每个用户的正样本(>=4)类型统计
        ratings = pd.read_csv(f"{self.data_path}/ratings.dat", sep='::', engine='python',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
        movies = pd.read_csv(f"{self.data_path}/movies.dat", sep='::', engine='python',
                             names=['movie_id', 'title', 'genres'], encoding='latin-1')
        movies, all_genres = self.processor.process_genres(movies)
        genre_cols = [f'genre_{g}' for g in all_genres]

        # 仅保留抽样用户的高分交互
        ratings_sub = ratings[ratings['user_id'].isin(sampled_raw)]
        ratings_pos = ratings_sub[ratings_sub['rating'] >= 4]
        user_genre_pref = ratings_pos.merge(movies[['movie_id'] + genre_cols], on='movie_id', how='left')
        # 按用户计算各类型的平均出现率（作为偏好近似）
        user_pref = user_genre_pref.groupby('user_id')[genre_cols].mean().fillna(0)

        # 组装分析矩阵
        emb_mat = user_emb_weight[sampled_idx, :top_dims]
        # 对齐索引
        common_users = np.intersect1d(user_pref.index.values, stats.index.values)
        if len(common_users) == 0:
            print("抽样用户过少或无交集，无法分析。")
            return
        emb_mat = emb_mat[np.isin(sampled_raw, common_users)]
        stats = stats.loc[common_users]
        user_pref = user_pref.loc[common_users]

        # 计算每个维度与统计特征/类型偏好的皮尔逊相关
        stat_cols = ['avg_rating', 'std_rating', 'rating_count', 'activity_days']
        report_lines = []
        for d in range(top_dims):
            dim_vec = emb_mat[:, d]
            # 与统计特征的相关
            stat_corr = {c: np.corrcoef(dim_vec, stats[c].values)[0,1] if len(stats[c].values)>1 else np.nan for c in stat_cols}
            # 与类型偏好的前三强相关
            genre_corrs = {g: np.corrcoef(dim_vec, user_pref[g].values)[0,1] if len(user_pref[g].values)>1 else np.nan for g in genre_cols}
            top_genres = sorted(genre_corrs.items(), key=lambda x: (np.nan_to_num(x[1], nan=0)), reverse=True)[:3]
            report_lines.append((d, stat_corr, top_genres))

        # 打印报告
        print("\n用户Embedding维度分析 (前 %d 维):" % top_dims)
        for d, stat_corr, top_genres in report_lines:
            print(f"维度 {d}: ")
            print("  与用户统计特征的相关:")
            for k, v in stat_corr.items():
                print(f"    - {k}: {v:.3f}")
            print("  与类型偏好的Top-3相关:")
            for g, v in top_genres:
                print(f"    - {g.replace('genre_','')}: {v:.3f}")
        print("\n提示: 相关系数越接近 ±1，说明该维与对应特征越相关。")

    def quick_probe_user_embedding(self, user_ids=None, top_dims=10):
        """便捷探针：一行调用完成Embedding维度报告"""
        self.analyze_user_embedding(user_ids=user_ids, top_dims=top_dims)
    
    def evaluate_ranking_metrics(self, test_ratings, k_list=[5, 10, 20]):
        """
        评估 Ranking 指标：Precision@K, Recall@K, NDCG@K
        
        参数:
            test_ratings: 测试集 DataFrame，包含 user_id, movie_id, rating 列
            k_list: 要评估的 K 值列表
        
        返回:
            各指标的平均值字典
        """
        print("\n" + "="*80)
        print("评估 Ranking 指标 (Precision@K, Recall@K, NDCG@K)")
        print("="*80)
        
        # 筛选正样本（评分 >= 4）
        test_positives = test_ratings[test_ratings['rating'] >= 4]
        
        # 按用户分组
        user_groups = test_positives.groupby('user_id')['movie_id'].apply(set).to_dict()
        
        metrics = {f'Precision@{k}': [] for k in k_list}
        metrics.update({f'Recall@{k}': [] for k in k_list})
        metrics.update({f'NDCG@{k}': [] for k in k_list})
        
        test_users = list(user_groups.keys())[:100]  # 测试前100个用户
        print(f"在 {len(test_users)} 个测试用户上评估...\n")
        
        for idx, user_id in enumerate(test_users):
            if (idx + 1) % 20 == 0:
                print(f"  已处理 {idx+1}/{len(test_users)} 用户...")
            
            try:
                # 生成推荐
                max_k = max(k_list)
                recommended_movies, scores = self.recommend(user_id, top_k=max_k)
                
                if len(recommended_movies) == 0:
                    continue
                
                # 获取用户的正样本
                true_items = user_groups.get(user_id, set())
                
                if len(true_items) == 0:
                    continue
                
                # 计算各种 K 值的指标
                for k in k_list:
                    pred_k = set(recommended_movies[:k])
                    hits = len(pred_k & true_items)
                    
                    # Precision@K
                    precision = hits / k if k > 0 else 0
                    metrics[f'Precision@{k}'].append(precision)
                    
                    # Recall@K
                    recall = hits / len(true_items) if len(true_items) > 0 else 0
                    metrics[f'Recall@{k}'].append(recall)
                    
                    # NDCG@K
                    dcg = sum([1 / np.log2(i + 2) if recommended_movies[i] in true_items else 0 
                              for i in range(min(k, len(recommended_movies)))])
                    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(true_items)))])
                    ndcg = dcg / idcg if idcg > 0 else 0
                    metrics[f'NDCG@{k}'].append(ndcg)
                    
            except Exception as e:
                print(f"  用户 {user_id} 评估失败: {e}")
                continue
        
        # 计算平均值
        results = {}
        print("\n" + "="*80)
        print("评估结果:")
        print("="*80)
        for metric_name, values in metrics.items():
            if len(values) > 0:
                avg_value = np.mean(values)
                results[metric_name] = avg_value
                print(f"{metric_name:20s}: {avg_value:.4f}")
            else:
                results[metric_name] = 0.0
                print(f"{metric_name:20s}: N/A")
        
        print("="*80)
        return results
