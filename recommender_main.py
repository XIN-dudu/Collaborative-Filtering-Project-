# recommender_main.py

import pandas as pd
import numpy as np
import UserMatrix2  # 你的 UserMatrix2 模块
import recommender_algorithms as ra  # 推荐算法模块
import recommender_eval as re  # 评估模块
import os  # 用于检查文件是否存在
import joblib  # 用于保存和加载对象
import pickle  # 用于保存和加载非 joblib 格式的对象，如 dict

# 从 sklearn 导入用于数据划分的工具
from sklearn.model_selection import train_test_split

# 定义数据文件路径
USER_ARTISTS_DAT_PATH = 'resources/user_artists.dat'
ARTISTS_DAT_PATH = 'resources/artists.dat'
USER_FRIENDS_DAT_PATH = 'resources/user_friends.dat'  # 新增：社交数据路径

# 定义缓存文件夹和文件路径
CACHE_DIR = 'cache'
# 注意：这里我们不再直接缓存 user_similarity_matrix.pkl 和 item_similarity_matrix.pkl，
# 而是会缓存融合后的用户相似度矩阵和纯余弦物品相似度矩阵
USER_FRIENDS_CACHE = os.path.join(CACHE_DIR, 'user_friends_data.pkl')  # 新增：社交数据缓存

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 定义过滤参数 (与 UserMatrix2 保持一致)
MIN_LISTEN_COUNT = 10
MIN_USERS_PER_ARTIST = 5
MIN_ARTISTS_PER_USER = 10

# 定义协同过滤参数
NUM_RECOMMENDATIONS = 10  # 推荐数量
K_NEIGHBORS = 50  # 邻居数量

# 融合社交信息的权重 (超参数，可以调整)
ALPHA_FUSION = 0.3  # 0表示纯行为相似度，1表示纯社交相似度

# 快速评估模式设置
FAST_EVAL_MODE = True
FAST_EVAL_USER_LIMIT = 200  # 限制评估用户数量

print("--- 音乐推荐系统启动 ---")

# --- 阶段 1: 数据加载与用户-艺术家矩阵构建 ---
print("\n阶段 1: 数据加载与用户-艺术家矩阵构建")

# 首次加载原始 user_artists.dat，用于后续的训练集/测试集划分
print("原始 user_artists.dat 加载中，用于划分训练集和测试集...")
try:
    raw_user_artists_df = pd.read_csv(USER_ARTISTS_DAT_PATH, sep='\t')
    print(f"原始 user_artists.dat 加载完成，共有 {len(raw_user_artists_df)} 条记录。")
except FileNotFoundError:
    print(f"错误: 未找到 {USER_ARTISTS_DAT_PATH}。请检查文件路径。")
    exit()

# 加载艺术家信息 (artist.dat)
print("加载 artist.dat 信息...")
try:
    artist_info_df = pd.read_csv(ARTISTS_DAT_PATH, sep='\t')
    print(f"artist.dat 加载完成，共有 {len(artist_info_df)} 条记录。")
except FileNotFoundError:
    print(f"错误: 未找到 {ARTISTS_DAT_PATH}。请检查文件路径。")
    exit()

# 划分训练集和测试集 (基于原始 df_user_artists)
train_df, test_df = train_test_split(raw_user_artists_df, test_size=0.2, random_state=42)

print(f"\n数据已划分为训练集 ({len(train_df)} 条记录) 和测试集 ({len(test_df)} 条记录)。")

# 使用训练集构建用户-艺术家CSR矩阵和映射
print("\n开始使用训练集构建用户-艺术家CSR矩阵和映射...")
user_artist_matrix_train, user_id_to_idx_train, idx_to_user_id_train, \
    artist_id_to_idx_train, idx_to_artist_id_train, user_item_ratings_train = \
    UserMatrix2.build_user_artist_matrix(
        user_artists_path=None,  # 从DataFrame构建
        artists_path=ARTISTS_DAT_PATH,  # 仅为函数兼容性保留，实际从input_df提取
        min_listen_count=MIN_LISTEN_COUNT,
        min_users_per_artist=MIN_USERS_PER_ARTIST,
        min_artists_per_user=MIN_ARTISTS_PER_USER,
        input_df=train_df
    )

if user_artist_matrix_train.shape[0] == 0:
    print("\n--- 阶段 1 失败：用户-艺术家训练矩阵构建失败或为空。 ---")
    print("请检查过滤条件或数据完整性。")
    exit()

print("\n--- 阶段 1 完成：用户-艺术家训练矩阵成功构建 ---")
print("训练矩阵维度:", user_artist_matrix_train.shape)
print(f"训练集用户数: {len(user_id_to_idx_train)}, 艺术家数: {len(artist_id_to_idx_train)}")

# 提取测试集的 user_item_ratings (不需要构建测试矩阵，只需要用于评估的字典)
# 注意：测试集的用户和艺术家也需要经过与训练集相同的过滤，以保证 ID 的一致性
print("\n准备测试集的用户-艺术家播放次数字典...")
_, _, _, _, _, user_item_ratings_test = \
    UserMatrix2.build_user_artist_matrix(
        user_artists_path=None,
        artists_path=ARTISTS_DAT_PATH,
        min_listen_count=MIN_LISTEN_COUNT,  # 测试集也应使用相同的过滤条件
        min_users_per_artist=MIN_USERS_PER_ARTIST,
        min_artists_per_user=MIN_ARTISTS_PER_USER,
        input_df=test_df
    )
print(f"测试集用户播放记录加载完成，包含 {len(user_item_ratings_test)} 个用户的记录。")

# --- 阶段 2: 加载社交数据 ---
print("\n阶段 2: 加载社交数据...")


def load_user_friends(filepath):
    """
    加载用户好友关系，并缓存。
    """
    if os.path.exists(USER_FRIENDS_CACHE):
        print(f"从缓存加载好友数据: {USER_FRIENDS_CACHE}")
        with open(USER_FRIENDS_CACHE, 'rb') as f:
            return pickle.load(f)

    user_friends = {}
    try:
        with open(filepath, 'r') as f:
            next(f)  # skip header
            for line in f:
                user1, user2 = map(int, line.strip().split('\t'))
                # 确保只加载在训练集中的用户的好友关系，减少内存占用和计算
                # 并且好友关系是双向的
                if user1 in user_id_to_idx_train and user2 in user_id_to_idx_train:
                    user_friends.setdefault(user1, set()).add(user2)
                    user_friends.setdefault(user2, set()).add(user1)
    except FileNotFoundError:
        print(f"错误: {filepath} 未找到。请确保 'resources/user_friends.dat' 存在。")
        return None

    print(f"加载 {len(user_friends)} 个用户的社交数据。")
    with open(USER_FRIENDS_CACHE, 'wb') as f:
        pickle.dump(user_friends, f)
    print(f"好友数据已缓存到 {USER_FRIENDS_CACHE}")
    return user_friends


user_friends_data = load_user_friends(USER_FRIENDS_DAT_PATH)
if user_friends_data is None:
    print("社交数据加载失败，将无法使用社交信息进行推荐。")
    user_friends_data = {}  # 设为空字典，防止后续报错

# --- 阶段 3: 协同过滤算法实现 - 相似度计算 (融合社交信息) ---
print("\n--- 阶段 3: 计算协同过滤相似度 (融合社交信息) ---")

user_similarity_matrix_fused = None  # 融合后的用户相似度矩阵

# 定义融合相似度矩阵的缓存路径，包含 ALPHA_FUSION 值
fused_sim_cache_path = os.path.join(CACHE_DIR, f'user_similarity_fused_alpha_{ALPHA_FUSION}.pkl')

# 尝试加载缓存的融合相似度矩阵
if os.path.exists(fused_sim_cache_path):
    print(f"\n尝试从缓存文件 '{fused_sim_cache_path}' 加载融合相似度矩阵...")
    try:
        user_similarity_matrix_fused = joblib.load(fused_sim_cache_path)
        print("融合相似度矩阵已从缓存加载。")
        # 简单检查形状是否匹配。更严格的检查需要对比用户ID映射，但通常形状匹配即可
        if user_similarity_matrix_fused.shape[0] != user_artist_matrix_train.shape[0] or \
                user_similarity_matrix_fused.shape[1] != user_artist_matrix_train.shape[0]:
            print("警告：缓存的融合相似度矩阵形状不匹配，将重新计算。")
            user_similarity_matrix_fused = None
    except Exception as e:
        print(f"加载缓存失败: {e}。将重新计算融合相似度矩阵。")
        user_similarity_matrix_fused = None

# 如果缓存不存在或加载失败，则重新计算
if user_similarity_matrix_fused is None:
    print("\n计算融合用户相似度矩阵（未命中缓存或缓存失效）...")
    user_similarity_matrix_fused = ra.calculate_user_similarity_social_fused(
        user_artist_matrix_train,
        user_friends_data,
        user_id_to_idx_train,
        idx_to_user_id_train,
        alpha=ALPHA_FUSION
    )
    joblib.dump(user_similarity_matrix_fused, fused_sim_cache_path)
    print(f"融合用户相似度矩阵已计算并保存到缓存文件 '{fused_sim_cache_path}'。")

print("融合用户相似度矩阵维度:", user_similarity_matrix_fused.shape)

# 为 Item-Based CF 计算物品相似度 (仍使用纯余弦相似度)
item_similarity_matrix = None
# 定义物品相似度矩阵的缓存路径
item_sim_cache_path = os.path.join(CACHE_DIR, 'item_similarity_matrix.pkl')

if os.path.exists(item_sim_cache_path):
    print(f"\n尝试从缓存文件 '{item_sim_cache_path}' 加载物品相似度矩阵...")
    try:
        item_similarity_matrix = joblib.load(item_sim_cache_path)
        if item_similarity_matrix.shape[0] != user_artist_matrix_train.shape[1] or \
                item_similarity_matrix.shape[1] != user_artist_matrix_train.shape[1]:
            print("警告：缓存的物品相似度矩阵形状不匹配，将重新计算。")
            item_similarity_matrix = None
    except Exception as e:
        print(f"加载缓存失败: {e}。将重新计算物品相似度矩阵。")
        item_similarity_matrix = None

if item_similarity_matrix is None:
    print("\n计算物品（艺术家）相似度...")
    artist_user_matrix_train = user_artist_matrix_train.T  # CSR matrix 的转置是 CSC matrix
    item_similarity_matrix = ra.calculate_item_similarity_cosine(artist_user_matrix_train)
    joblib.dump(item_similarity_matrix, item_sim_cache_path)
    print(f"物品相似度矩阵已计算并保存到缓存文件 '{item_sim_cache_path}'。")

print("物品相似度矩阵维度:", item_similarity_matrix.shape)

# --- 阶段 4: 推荐生成和评估 ---
print("\n--- 阶段 4: 推荐生成和评估 ---")

# --- 4.1 演示单个用户的推荐 (使用 User-Based CF 融合社交信息) ---
# 找一个存在于训练集中的用户ID作为示例
if len(user_id_to_idx_train) > 0:
    example_user_id = next(iter(user_id_to_idx_train.keys()))  # 获取第一个用户ID
else:
    example_user_id = -1

print(f"\n为用户 {example_user_id} 生成基于用户协同过滤 (融合社交信息) 的 Top-{NUM_RECOMMENDATIONS} 推荐：")
if user_similarity_matrix_fused is not None:
    user_based_recommended_artist_ids = ra.recommend_user_based_cf(
        user_id=example_user_id,
        user_artist_matrix=user_artist_matrix_train,  # 传递原始矩阵，如果算法需要
        user_similarity_matrix=user_similarity_matrix_fused,  # 使用融合后的相似度
        user_id_to_idx=user_id_to_idx_train,
        idx_to_user_id=idx_to_user_id_train,
        user_item_ratings_train=user_item_ratings_train,  # 传递训练集播放记录用于过滤
        num_recommendations=NUM_RECOMMENDATIONS,
        k_neighbors=K_NEIGHBORS
    )
    # 将推荐的艺术家ID转换为名称以便显示
    recommended_artist_names = []
    for artist_id in user_based_recommended_artist_ids:
        artist_name_row = artist_info_df.loc[artist_info_df['id'] == artist_id, 'name']
        artist_name = artist_name_row.iloc[0] if not artist_name_row.empty else f"未知艺术家 ({artist_id})"
        recommended_artist_names.append(artist_name)

    print(f"为用户 {example_user_id} 推荐的艺术家：")
    if recommended_artist_names:
        for i, artist_name in enumerate(recommended_artist_names):
            print(f"{i + 1}. {artist_name}")
    else:
        print("没有生成推荐。")
else:
    print("融合用户相似度矩阵为空，无法进行用户推荐示例。")

# --- 4.2 演示单个用户的推荐 (使用 Item-Based CF) ---
print(f"\n为用户 {example_user_id} 生成基于物品协同过滤的 Top-{NUM_RECOMMENDATIONS} 推荐：")
if item_similarity_matrix is not None:
    item_based_recommended_artist_ids = ra.recommend_item_based_cf(
        user_id=example_user_id,
        user_artist_matrix=user_artist_matrix_train,  # 传递原始矩阵，如果算法需要
        item_similarity_matrix=item_similarity_matrix,
        user_id_to_idx=user_id_to_idx_train,
        artist_id_to_idx=artist_id_to_idx_train,
        idx_to_artist_id=idx_to_artist_id_train,
        user_item_ratings_train=user_item_ratings_train,  # 传递训练集播放记录用于过滤
        num_recommendations=NUM_RECOMMENDATIONS,
        k_neighbors=K_NEIGHBORS
    )
    # 将推荐的艺术家ID转换为名称以便显示
    recommended_artist_names_item_based = []
    for artist_id in item_based_recommended_artist_ids:
        artist_name_row = artist_info_df.loc[artist_info_df['id'] == artist_id, 'name']
        artist_name = artist_name_row.iloc[0] if not artist_name_row.empty else f"未知艺术家 ({artist_id})"
        recommended_artist_names_item_based.append(artist_name)

    print(f"为用户 {example_user_id} 推荐的艺术家：")
    if recommended_artist_names_item_based:
        for i, artist_name in enumerate(recommended_artist_names_item_based):
            print(f"{i + 1}. {artist_name}")
    else:
        print("没有生成推荐。")
else:
    print("物品相似度矩阵为空，无法进行物品推荐示例。")

# --- 4.3 评估推荐系统 ---
print("\n--- 评估推荐系统性能 ---")

# 评估 User-Based CF (融合社交信息)
if user_similarity_matrix_fused is not None:
    print(f"\n开始评估 User-Based CF (融合社交信息，快速模式={FAST_EVAL_MODE})...")

    # 获取训练集和测试集中都存在的用户
    users_in_train_and_test = set(user_id_to_idx_train.keys()).intersection(user_item_ratings_test.keys())

    users_to_evaluate_ids_raw = list(users_in_train_and_test)

    users_for_eval = []
    if FAST_EVAL_MODE:
        # 随机选择一部分用户进行评估
        np.random.seed(42)  # 保证每次运行选到的用户相同
        # 从 all_test_users 中抽取
        users_for_eval = np.random.choice(users_to_evaluate_ids_raw,
                                          min(FAST_EVAL_USER_LIMIT, len(users_to_evaluate_ids_raw)),
                                          replace=False).tolist()
        print(f"!!! 快速评估模式：只评估 {len(users_for_eval)} 个用户 !!!")
    else:
        users_for_eval = users_to_evaluate_ids_raw

    # 进一步过滤测试集，只保留那些在训练矩阵中存在的用户和艺术家
    # 并且只考虑在快速评估模式下选择的用户
    eval_test_ratings_filtered = {}
    for user_id in users_for_eval:
        # 确保测试集中用户的艺术家也在训练矩阵中存在
        current_user_test_artists = user_item_ratings_test.get(user_id, {})
        filtered_artists = {
            artist_id: count for artist_id, count in current_user_test_artists.items()
            if artist_id in artist_id_to_idx_train  # 艺术家在训练集构建的矩阵中存在
        }
        if filtered_artists:  # 如果过滤后用户仍有有效测试记录
            eval_test_ratings_filtered[user_id] = filtered_artists

    print(f"最终用于评估的用户数量: {len(eval_test_ratings_filtered)}")

    if eval_test_ratings_filtered:
        avg_precision_ub, avg_recall_ub = re.evaluate_model(
            recommendation_function=lambda uid: ra.recommend_user_based_cf(
                user_id=uid,
                user_artist_matrix=user_artist_matrix_train,
                user_similarity_matrix=user_similarity_matrix_fused,
                user_id_to_idx=user_id_to_idx_train,
                idx_to_user_id=idx_to_user_id_train,
                user_item_ratings_train=user_item_ratings_train,
                num_recommendations=NUM_RECOMMENDATIONS,
                k_neighbors=K_NEIGHBORS
            ),
            user_item_ratings_test=eval_test_ratings_filtered,  # 使用过滤后的测试集
            users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),  # 传递实际要评估的用户ID列表
            top_k=NUM_RECOMMENDATIONS
        )
        print(f"User-Based CF (Social Fused) 平均 Precision@{NUM_RECOMMENDATIONS}: {avg_precision_ub:.4f}")
        print(f"User-Based CF (Social Fused) 平均 Recall@{NUM_RECOMMENDATIONS}: {avg_recall_ub:.4f}")
    else:
        print("没有可用于评估 User-Based CF 的有效测试用户。")

else:
    print("融合用户相似度矩阵为空，跳过 User-Based CF 评估。")

# 评估 Item-Based CF
if item_similarity_matrix is not None:
    print(f"\n开始评估 Item-Based CF (快速模式={FAST_EVAL_MODE})...")
    if eval_test_ratings_filtered:  # 使用与 User-Based CF 相同的过滤用户列表进行评估
        avg_precision_ib, avg_recall_ib = re.evaluate_model(
            recommendation_function=lambda uid: ra.recommend_item_based_cf(
                user_id=uid,
                user_artist_matrix=user_artist_matrix_train,
                item_similarity_matrix=item_similarity_matrix,
                user_id_to_idx=user_id_to_idx_train,
                artist_id_to_idx=artist_id_to_idx_train,
                idx_to_artist_id=idx_to_artist_id_train,
                user_item_ratings_train=user_item_ratings_train,
                num_recommendations=NUM_RECOMMENDATIONS,
                k_neighbors=K_NEIGHBORS
            ),
            user_item_ratings_test=eval_test_ratings_filtered,  # 使用过滤后的测试集
            users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),  # 传递实际要评估的用户ID列表
            top_k=NUM_RECOMMENDATIONS
        )
        print(f"Item-Based CF 平均 Precision@{NUM_RECOMMENDATIONS}: {avg_precision_ib:.4f}")
        print(f"Item-Based CF 平均 Recall@{NUM_RECOMMENDATIONS}: {avg_recall_ib:.4f}")
    else:
        print("没有可用于评估 Item-Based CF 的有效测试用户。")
else:
    print("物品相似度矩阵为空，跳过 Item-Based CF 评估。")

print("\n--- 音乐推荐系统关闭 ---")