"""
快速测试 Wide & Deep 推荐系统
"""

from wide_deep_recommender import MovieRecommender
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("Wide & Deep 电影推荐系统 - 快速测试")
    print("="*80)
    
    # 初始化推荐系统
    DATA_PATH = './data/ml-10M100K'
    recommender = MovieRecommender(DATA_PATH)
    
    # 准备数据
    print("\n步骤 1: 加载和处理数据...")
    ratings, movies, train_data, user_stats, movie_features, all_genres = recommender.prepare_data()
    
    print(f"\n数据概览:")
    print(f"  - 用户数: {ratings['user_id'].nunique():,}")
    print(f"  - 电影数: {movies['movie_id'].nunique():,}")
    print(f"  - 评分数: {len(ratings):,}")
    print(f"  - 训练样本数: {len(train_data):,}")
    
    # 构建和训练模型（使用较少的 epochs 进行快速测试）
    print("\n步骤 2: 构建和训练 Wide & Deep 模型...")
    print("注意: 使用 3 个 epochs 进行快速测试，实际应用建议使用 10+ epochs")
    
    history = recommender.build_and_train(
        train_data, 
        user_stats, 
        movie_features, 
        all_genres,
        epochs=3,
        batch_size=2048
    )
    
    print(f"\n训练完成!")
    print(f"  - 验证集 AUC: {history.history['val_auc'][-1]:.4f}")
    print(f"  - 验证集准确率: {history.history['val_accuracy'][-1]:.4f}")
    
    # 设置推荐引擎
    print("\n步骤 3: 设置召回和重排序引擎...")
    recommender.setup_engines(ratings, movies, movie_features, user_stats, all_genres)
    
    # 测试推荐
    print("\n步骤 4: 生成推荐...")
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
        ).sort_values('rating', ascending=False).head(5)
        
        print(f"\n用户历史偏好 (Top 5):")
        for idx, row in user_history.iterrows():
            print(f"  {row['title'][:50]:50s} | 类型: {row['genres']:30s} | 评分: {row['rating']}")
        
        # 生成推荐
        try:
            rec_movies, rec_scores = recommender.recommend(user_id, top_k=10)
            
            print(f"\n推荐结果 (Top 10):")
            for i, (movie_id, score) in enumerate(zip(rec_movies, rec_scores), 1):
                movie = movies[movies['movie_id'] == movie_id].iloc[0]
                print(f"  {i:2d}. {movie['title'][:50]:50s} | 类型: {movie['genres']:30s} | 预测: {score:.4f}")
        except Exception as e:
            print(f"  推荐失败: {e}")
    
    # 保存模型
    print(f"\n{'='*80}")
    print("步骤 5: 保存模型...")
    recommender.save(
        model_path='./wide_deep_model.pth',
        processor_path='./processor.pkl'
    )
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print("\n使用建议:")
    print("1. 在 Jupyter Notebook 中运行 Wide_Deep_Movie_Recommender.ipynb 查看完整流程")
    print("2. 调整 epochs 参数（10-20）以获得更好的推荐效果")
    print("3. 可以添加更多特征来提升模型性能")
    print("4. 尝试调整召回策略和重排序参数")
    

if __name__ == '__main__':
    main()
