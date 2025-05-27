import pandas as pd
import numpy as np
import UserMatrix2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split # 引入用于数据划分的工具

# 导入我们刚刚创建的模块
import recommender_algorithms as ra # 推荐算法模块
import recommender_eval as re       # 评估模块

# 定义数据文件路径
USER_ARTISTS_DAT_PATH = 'resources/user_artists.dat'
ARTISTS_DAT_PATH = 'resources/artists.dat'

print("--- 音乐推荐系统启动 ---")

# 阶段 1: 数据加载与用户-艺术家矩阵构建
print("阶段 1: 数据加载与用户-艺术家矩阵构建")

# 首次加载原始 user_artists.dat，用于后续的训练集/测试集划分
print("原始 user_artists.dat 加载中，用于划分训练集和测试集...")
raw_user_artists_df = pd.read_csv(USER_ARTISTS_DAT_PATH, sep='\t')
print(f"原始 user_artists.dat 加载完成，共有 {len(raw_user_artists_df)} 条记录。")

# 划分训练集和测试集
train_df, test_df = train_test_split(raw_user_artists_df, test_size=0.2, random_state=42) # 80% 训练，20% 测试

print(f"\n数据已划分为训练集 ({len(train_df)} 条记录) 和测试集 ({len(test_df)} 条记录)。")

# 使用训练集构建用户-艺术家矩阵
print("\n开始使用训练集构建用户-艺术家矩阵...")
user_artist_matrix_train, artist_info_df = UserMatrix2.build_user_artist_matrix(
    user_artists_path=None, # 不再从文件加载，而是从 DataFrame
    artists_path=ARTISTS_DAT_PATH,
    min_listen_count=10,
    min_users_per_artist=5,
    min_artists_per_user=10,
    input_df=train_df # 传入训练集 DataFrame
)

if not user_artist_matrix_train.empty:
    print("\n--- 阶段 1 完成：用户-艺术家训练矩阵成功构建 ---")
    print("训练矩阵维度:", user_artist_matrix_train.shape)
    print("训练矩阵示例（前5行5列）：")
    print(user_artist_matrix_train.iloc[:5, :5])

    # --- 阶段 2: 协同过滤算法实现 - 相似度计算 ---
    print("\n--- 阶段 2: 计算协同过滤相似度 ---")

    print("\n--- 计算用户相似度（User-Based CF） ---")
    try:
        user_similarity_array = cosine_similarity(user_artist_matrix_train.values)
        user_similarity_matrix = pd.DataFrame(
            user_similarity_array,
            index=user_artist_matrix_train.index,
            columns=user_artist_matrix_train.index
        )
        print("\n用户相似度矩阵计算完成。")
        print("用户相似度矩阵维度:", user_similarity_matrix.shape)

    except Exception as e:
        print(f"\n计算用户相似度时发生错误: {e}")
        user_similarity_matrix = pd.DataFrame() # 错误时置为空，避免后续报错

    print("\n\n--- 计算物品（艺术家）相似度（Item-Based CF） ---")
    artist_user_matrix_train = user_artist_matrix_train.T # 转置矩阵

    try:
        item_similarity_array = cosine_similarity(artist_user_matrix_train.values)
        item_similarity_matrix = pd.DataFrame(
            item_similarity_array,
            index=artist_user_matrix_train.index,
            columns=artist_user_matrix_train.index
        )
        print("\n物品（艺术家）相似度矩阵计算完成。")
        print("物品相似度矩阵维度:", item_similarity_matrix.shape)

    except Exception as e:
        print(f"\n计算物品相似度时发生错误: {e}")
        item_similarity_matrix = pd.DataFrame() # 错误时置为空，避免后续报错

    # --- 阶段 4: 推荐生成和评估 ---
    print("\n--- 阶段 4: 推荐生成和评估 ---")

    if not user_artist_matrix_train.index.empty and not artist_info_df.empty:
        # --- 4.1 演示单个用户的推荐 (使用 User-Based CF) ---
        example_user_id = user_artist_matrix_train.index[0]
        print(f"\n为用户 {example_user_id} 生成基于用户协同过滤的 Top-10 推荐：")
        if not user_similarity_matrix.empty:
            user_based_recommended_artist_ids = ra.recommend_user_based_cf(
                user_id=example_user_id,
                user_artist_matrix=user_artist_matrix_train,
                user_similarity_matrix=user_similarity_matrix,
                num_recommendations=10
            )
            # 将推荐的艺术家ID转换为名称以便显示
            recommended_artist_names = []
            for artist_id in user_based_recommended_artist_ids:
                artist_name_row = artist_info_df.loc[artist_info_df['artistID'] == artist_id, 'artistName']
                artist_name = artist_name_row.iloc[0] if not artist_name_row.empty else f"未知艺术家 ({artist_id})"
                recommended_artist_names.append(artist_name)

            for i, artist_name in enumerate(recommended_artist_names):
                print(f"{i+1}. {artist_name}")
        else:
            print("用户相似度矩阵为空，无法进行用户推荐示例。")

        # --- 4.2 演示单个用户的推荐 (使用 Item-Based CF) ---
        print(f"\n为用户 {example_user_id} 生成基于物品协同过滤的 Top-10 推荐：")
        if not item_similarity_matrix.empty:
            item_based_recommended_artist_ids = ra.recommend_item_based_cf(
                user_id=example_user_id,
                user_artist_matrix=user_artist_matrix_train,
                item_similarity_matrix=item_similarity_matrix,
                num_recommendations=10
            )
            # 将推荐的艺术家ID转换为名称以便显示
            recommended_artist_names_item_based = []
            for artist_id in item_based_recommended_artist_ids:
                artist_name_row = artist_info_df.loc[artist_info_df['artistID'] == artist_id, 'artistName']
                artist_name = artist_name_row.iloc[0] if not artist_name_row.empty else f"未知艺术家 ({artist_id})"
                recommended_artist_names_item_based.append(artist_name)

            for i, artist_name in enumerate(recommended_artist_names_item_based):
                print(f"{i+1}. {artist_name}")
        else:
            print("物品相似度矩阵为空，无法进行物品推荐示例。")

        # --- 4.3 评估推荐系统 ---
        print("\n--- 评估推荐系统性能 ---")

        # 评估 User-Based CF
        if not user_similarity_matrix.empty:
            print("\n开始评估 User-Based CF...")
            avg_precision_ub, avg_recall_ub = re.evaluate_model(
                recommendation_function=ra.recommend_user_based_cf,
                test_df=test_df,
                user_artist_matrix_train=user_artist_matrix_train,
                user_similarity_matrix=user_similarity_matrix,
                num_recommendations=10
            )
            print(f"User-Based CF 平均 Precision@10: {avg_precision_ub:.4f}")
            print(f"User-Based CF 平均 Recall@10: {avg_recall_ub:.4f}")
        else:
            print("用户相似度矩阵为空，跳过 User-Based CF 评估。")

        # 评估 Item-Based CF
        if not item_similarity_matrix.empty:
            print("\n开始评估 Item-Based CF...")
            avg_precision_ib, avg_recall_ib = re.evaluate_model(
                recommendation_function=ra.recommend_item_based_cf,
                test_df=test_df,
                user_artist_matrix_train=user_artist_matrix_train,
                item_similarity_matrix=item_similarity_matrix,
                artist_info_df=artist_info_df, # 传递 artist_info_df
                num_recommendations=10
            )
            print(f"Item-Based CF 平均 Precision@10: {avg_precision_ib:.4f}")
            print(f"Item-Based CF 平均 Recall@10: {avg_recall_ib:.4f}")
        else:
            print("物品相似度矩阵为空，跳过 Item-Based CF 评估。")

    else:
        print("矩阵或艺术家信息不完整，无法进行推荐生成和评估。")

else:
    print("\n--- 阶段 1 失败：用户-艺术家训练矩阵构建失败 ---")
    print("请检查文件路径和数据格式，或查看 UserMatrix2.py 的错误信息。")

print("\n--- 音乐推荐系统关闭 ---")