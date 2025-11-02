"""
Kaggle 环境下运行 Wide & Deep 推荐系统
直接运行: !python run_kaggle.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wide_deep_recommender import MovieRecommender

def main():
    print("="*80)
    print("Wide & Deep 电影推荐系统 - Kaggle 版本")
    print("="*80)
    
    # Kaggle 数据路径
    DATA_PATH = '/kaggle/input/ml-10m/ml-10M100K'
    
    # 1. 初始化推荐系统
    print("\n步骤 1: 初始化推荐系统...")
    recommender = MovieRecommender(DATA_PATH)
    
    # 2. 准备数据
    print("\n步骤 2: 加载和处理数据...")
    ratings, movies, train_data, user_stats, movie_features, all_genres = recommender.prepare_data()
    
    print(f"\n数据概览:")
    print(f"  用户数量: {ratings['user_id'].nunique():,}")
    print(f"  电影数量: {movies['movie_id'].nunique():,}")
    print(f"  评分数量: {len(ratings):,}")
    print(f"  训练样本: {len(train_data):,}")
    
    # 3. 训练模型
    print("\n步骤 3: 训练 Wide & Deep 模型...")
    history = recommender.build_and_train(
        train_data, 
        user_stats, 
        movie_features, 
        all_genres,
        epochs=5,
        batch_size=2048
    )
    
    print(f"\n训练完成!")
    print(f"  验证集 Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  验证集 Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  验证集 AUC: {history.history['val_auc'][-1]:.4f}")
    
    # 4. 可视化训练结果
    print("\n步骤 4: 可视化训练过程...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history.history['loss'], label='训练损失', marker='o')
    axes[0].plot(history.history['val_loss'], label='验证损失', marker='s')
    axes[0].set_title('模型损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='训练准确率', marker='o')
    axes[1].plot(history.history['val_accuracy'], label='验证准确率', marker='s')
    axes[1].set_title('模型准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history.history['auc'], label='训练 AUC', marker='o')
    axes[2].plot(history.history['val_auc'], label='验证 AUC', marker='s')
    axes[2].set_title('模型 AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("  训练曲线已保存: training_history.png")
    plt.show()
    
    # 5. 设置推荐引擎
    print("\n步骤 5: 设置推荐引擎...")
    recommender.setup_engines(ratings, movies, movie_features, user_stats, all_genres)
    
    # 6. 生成推荐
    print("\n步骤 6: 生成推荐...")
    test_users = [1, 100, 500]
    
    for user_id in test_users:
        if user_id not in ratings['user_id'].values:
            continue
            
        print(f"\n{'='*80}")
        print(f"为用户 {user_id} 生成推荐")
        print('='*80)
        
        # 用户历史
        user_history = ratings[ratings['user_id'] == user_id].merge(
            movies, on='movie_id'
        ).sort_values('rating', ascending=False).head(3)
        
        print(f"\n用户历史偏好 (Top 3):")
        for idx, row in user_history.iterrows():
            print(f"  {row['title'][:50]:50s} - 评分: {row['rating']}")
        
        # 生成推荐
        recommended_movies, scores = recommender.recommend(user_id, top_k=10)
        
        print(f"\n推荐结果 (Top 10):")
        for i, (movie_id, score) in enumerate(zip(recommended_movies, scores), 1):
            movie = movies[movies['movie_id'] == movie_id].iloc[0]
            print(f"  {i:2d}. {movie['title'][:50]:50s} - 预测: {score:.4f}")
    
    # 7. 保存模型
    print(f"\n{'='*80}")
    print("步骤 7: 保存模型...")
    recommender.save(
        model_path='wide_deep_model.pth',
        processor_path='processor.pkl'
    )
    
    print("\n" + "="*80)
    print("全部完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  - wide_deep_model.pth  (训练好的模型)")
    print("  - processor.pkl        (数据处理器)")
    print("  - training_history.png (训练曲线图)")
    

if __name__ == '__main__':
    main()
