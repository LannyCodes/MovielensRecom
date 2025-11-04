"""
Kaggle 环境下运行 Wide & Deep 推荐系统
使用方式:
  - 重新训练: !python run_kaggle.py
  - 加载模型: !python run_kaggle.py --load
  - 指定路径: !python run_kaggle.py --load --model-path /path/to/model.pth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from wide_deep_recommender import MovieRecommender

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Wide & Deep 电影推荐系统 - Kaggle版')
    parser.add_argument('--load', action='store_true', 
                       help='加载已有模型而非重新训练')
    parser.add_argument('--model-path', type=str, default='wide_deep_model.pth',
                       help='模型文件路径 (默认: wide_deep_model.pth)')
    parser.add_argument('--processor-path', type=str, default='processor.pkl',
                       help='处理器文件路径 (默认: processor.pkl)')
    parser.add_argument('--data-path', type=str, default='/kaggle/input/ml-10m/ml-10M100K',
                       help='数据路径 (默认: Kaggle路径)')
    args = parser.parse_args()
    
    print("="*80)
    print("Wide & Deep 电影推荐系统 - Kaggle 版本")
    print("="*80)
    
    # Kaggle 数据路径
    DATA_PATH = args.data_path
    
    # 1. 初始化推荐系统
    print("\n步骤 1: 初始化推荐系统...")
    recommender = MovieRecommender(DATA_PATH)
    
    # 检查是否加载已有模型
    if args.load:
        if not os.path.exists(args.model_path) or not os.path.exists(args.processor_path):
            print(f"\n⚠️  警告: 模型文件不存在!")
            print(f"  模型路径: {args.model_path} - {'存在' if os.path.exists(args.model_path) else '不存在'}")
            print(f"  处理器路径: {args.processor_path} - {'存在' if os.path.exists(args.processor_path) else '不存在'}")
            print(f"\n将改为重新训练模式...\n")
            args.load = False
        else:
            print(f"\n📦 加载已有模型:")
            print(f"  模型: {args.model_path}")
            print(f"  处理器: {args.processor_path}")
            recommender.load(args.model_path, args.processor_path)
            
            # 加载数据用于推荐引擎
            print("\n步骤 2: 加载数据用于推荐...")
            ratings, movies, _, user_stats, movie_features, all_genres = recommender.prepare_data()
            
            print(f"\n数据概览:")
            print(f"  用户数量: {ratings['user_id'].nunique():,}")
            print(f"  电影数量: {movies['movie_id'].nunique():,}")
            print(f"  评分数量: {len(ratings):,}")
            
            # 直接跳到设置推荐引擎
            print("\n步骤 3: 设置推荐引擎...")
            recommender.setup_engines(ratings, movies, movie_features, user_stats, all_genres)
    
    # 重新训练模式
    if not args.load:
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
            epochs=3,        # ⚡ 减少到 3 轮
            batch_size=4096  # ⚡ 增大 batch_size 减少内存占用
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
    step_num = 4 if args.load else 6
    print(f"\n步骤 {step_num}: 生成推荐...")
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
    
    # 7. 保存模型 (仅训练模式)
    if not args.load:
        step_num = 7
        print(f"\n{'='*80}")
        print(f"步骤 {step_num}: 保存模型...")
        recommender.save(
            model_path=args.model_path,
            processor_path=args.processor_path
        )
    
    print("\n" + "="*80)
    print("全部完成！")
    print("="*80)
    
    if not args.load:
        print("\n生成的文件:")
        print(f"  - {args.model_path}  (训练好的模型)")
        print(f"  - {args.processor_path}  (数据处理器)")
        print("  - training_history.png (训练曲线图)")
    else:
        print("\n使用的文件:")
        print(f"  - {args.model_path}  (已加载的模型)")
        print(f"  - {args.processor_path}  (已加载的处理器)")
    

if __name__ == '__main__':
    main()
