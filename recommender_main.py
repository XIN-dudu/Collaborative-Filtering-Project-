import pandas as pd
import numpy as np
import UserMatrix2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split # 引入用于数据划分的工具

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
# 这里我们对原始交互数据进行划分，而不是对构建好的矩阵进行划分
# 因为对矩阵直接划分可能导致某些用户或物品在训练集和测试集中都不出现。
# 更健壮的方法是按用户或按交互记录进行划分。
# 简单起见，我们对所有记录进行随机划分
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
    # user_artist_matrix_train 的行是用户，列是艺术家。
    try:
        user_similarity_array = cosine_similarity(user_artist_matrix_train.values)
        user_similarity_matrix = pd.DataFrame(
            user_similarity_array,
            index=user_artist_matrix_train.index,
            columns=user_artist_matrix_train.index
        )
        print("\n用户相似度矩阵计算完成。")
        print("用户相似度矩阵维度:", user_similarity_matrix.shape)
        print("用户相似度矩阵示例（前5行5列）：")
        print(user_similarity_matrix.iloc[:5, :5])

        if not user_artist_matrix_train.index.empty:
            target_user_id = user_artist_matrix_train.index[0]
            print(f"\n为用户 {target_user_id} 找到最相似的用户 (排除自己):")
            similar_users = user_similarity_matrix[target_user_id].drop(target_user_id, errors='ignore').sort_values(ascending=False)
            print(similar_users.head(5))
        else:
            print("用户索引为空，无法演示相似用户。")

    except Exception as e:
        print(f"\n计算用户相似度时发生错误: {e}")
        print("请确保 sklearn 已安装且 UserMatrix2.py 中的矩阵构建正确。")


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
        print("物品相似度矩阵示例（前5行5列）：")
        print(item_similarity_matrix.iloc[:5, :5])

        if not artist_user_matrix_train.index.empty:
            target_artist_id = artist_user_matrix_train.index[0]
            print(f"\n为艺术家 {target_artist_id} 找到最相似的艺术家 (排除自己):")
            similar_artists = item_similarity_matrix[target_artist_id].drop(target_artist_id, errors='ignore').sort_values(ascending=False)
            print(similar_artists.head(5))

            if not artist_info_df.empty:
                print("\n最相似艺术家的名称:")
                for artist_id_val, similarity_score in similar_artists.head(5).items():
                    artist_name_row = artist_info_df.loc[artist_info_df['artistID'] == artist_id_val, 'artistName']
                    artist_name = artist_name_row.iloc[0] if not artist_name_row.empty else f"未知艺术家 ({artist_id_val})"
                    print(f"- {artist_name} (相似度: {similarity_score:.4f})")
            else:
                print("艺术家信息 DataFrame 为空，无法显示艺术家名称。")
        else:
            print("艺术家索引为空，无法演示相似艺术家。")

    except Exception as e:
        print(f"\n计算物品相似度时发生错误: {e}")
        print("请确保 sklearn 已安装且 UserMatrix2.py 中的矩阵构建正确。")

    # --- 阶段 4: 推荐生成和评估 ---
    # 下一步我们将在这里实现推荐生成和评估的逻辑。
    print("\n--- 准备进入阶段 4: 推荐生成和评估 ---")

    # TODO: 在这里添加推荐生成和评估的函数调用

else:
    print("\n--- 阶段 1 失败：用户-艺术家训练矩阵构建失败 ---")
    print("请检查文件路径和数据格式，或查看 UserMatrix2.py 的错误信息。")

print("\n--- 音乐推荐系统关闭 ---")