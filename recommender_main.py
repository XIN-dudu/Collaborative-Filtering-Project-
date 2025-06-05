# recommender_main.py

import pandas as pd
import numpy as np
import UserMatrix2  # 你的 UserMatrix2 模块
import recommender_algorithms as ra  # 推荐算法模块
import recommender_eval as re  # 评估模块
import os  # 用于检查文件是否存在
import joblib  # 用于保存和加载对象
import pickle  # 用于保存和加载非 joblib 格式的对象，如 dict
import time  # 用于计时
from collections import defaultdict  # 用于统计标签频率

# 从 sklearn 导入用于数据划分的工具
from sklearn.model_selection import train_test_split

# 定义数据文件路径
USER_ARTISTS_DAT_PATH = 'resources/user_artists.dat'
ARTISTS_DAT_PATH = 'resources/artists.dat'
USER_FRIENDS_DAT_PATH = 'resources/user_friends.dat'  # 社交数据路径
# 新增：标签数据文件路径
TAGS_DAT_PATH = 'resources/tags.dat'
USER_TAGGED_ARTISTS_DAT_PATH = 'resources/user_taggedartists.dat'

# 定义缓存文件夹和文件路径
CACHE_DIR = 'cache'
USER_FRIENDS_CACHE = os.path.join(CACHE_DIR, 'user_friends_data.pkl')  # 社交数据缓存
ITEM_SIMILARITY_CACHE = os.path.join(CACHE_DIR, 'item_similarity_matrix.pkl')  # Item-Based CF 相似度矩阵缓存
# 新增：标签相关数据缓存路径 (如果未来需要缓存预处理结果)
ARTIST_TO_TAGS_CACHE = os.path.join(CACHE_DIR, 'artist_to_tags.pkl')
UNIQUE_TAGS_CACHE = os.path.join(CACHE_DIR, 'unique_tags.pkl')
# 注意：标签频率过滤的缓存文件名现在包含了过滤参数，以避免混淆
# 旧的 FILTERED_ARTIST_TO_TAGS_CACHE 和 FILTERED_UNIQUE_TAGS_CACHE 可以移除或根据需要更新
# FILTERED_ARTIST_TO_TAGS_CACHE = os.path.join(CACHE_DIR, 'filtered_artist_to_tags.pkl')
# FILTERED_UNIQUE_TAGS_CACHE = os.path.join(CACHE_DIR, 'filtered_unique_tags.pkl')


# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 定义过滤参数 (与 UserMatrix2 保持一致)
MIN_LISTEN_COUNT = 10
MIN_USERS_PER_ARTIST = 5
MIN_ARTISTS_PER_USER = 10

# --- 最终选择的最佳协同过滤参数 ---
# User-Based CF with Social Fusion 的最佳参数
BEST_ALPHA_FUSION_UB = 0.18
BEST_K_NEIGHBORS_UB = 200

# Item-Based CF 的最佳参数
BEST_K_NEIGHBORS_IB = 180

# 新增：混合推荐的最佳参数 (Content-Based 权重) - 使用上次调参的最佳结果
BEST_CB_WEIGHT_HYBRID = 0.09  # 根据上次调参结果
BEST_MIN_TAGS_PER_ARTIST_CB = 3  # 根据上次调参结果

# 新增：标签过滤参数 (用于 Content-Based 部分的全局过滤)
MIN_TAG_FREQ_GLOBAL = 5  # 标签在所有艺术家中出现的最低次数
MAX_TAG_FREQ_RATIO_GLOBAL = 0.5  # 标签在所有艺术家中出现的最高比例 (例如，0.5表示最多出现在50%的艺术家中)

NUM_RECOMMENDATIONS = 10  # 推荐数量

# --- 评估模式设置 ---
# 设置为 False 进行完整评估，这将评估所有符合条件的用户
FAST_EVAL_MODE = False
FAST_EVAL_USER_LIMIT = 200  # 快速评估模式下评估的用户数量 (FAST_EVAL_MODE=False 时此值无效)

print("--- 音乐推荐系统启动 ---")

# --- 阶段 1: 数据加载与用户-艺术家矩阵构建 ---
print("\n阶段 1: 数据加载与用户-艺术家矩阵构建")

# 首次加载原始 user_artists.dat，用于后续的训练集/测试集划分
print("原始 user_artists.dat 加载中，用于划分训练集和测试集...")
try:
    # 保持 user_artists.dat 和 artists.dat 默认 UTF-8 编码，通常它们是纯数字ID和英文名
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

# 新增：加载标签信息 (tags.dat)
print("加载 tags.dat 信息...")
try:
    # 重点修改：指定编码为 'latin1'
    tags_df = pd.read_csv(TAGS_DAT_PATH, sep='\t', encoding='latin1')
    # 创建 tagID 到 tagValue 的映射字典
    tag_id_to_value = dict(zip(tags_df['tagID'], tags_df['tagValue']))
    print(f"tags.dat 加载完成，共有 {len(tags_df)} 条记录。")
except FileNotFoundError:
    print(f"错误: 未找到 {TAGS_DAT_PATH}。请检查文件路径。")
    exit()
except UnicodeDecodeError as e:
    print(f"解码错误: 无法使用 latin1 编码读取 {TAGS_DAT_PATH}。尝试其他编码或检查文件。错误信息: {e}")
    exit()

# 新增：加载用户-艺术家标签数据 (user_taggedartists.dat)
print("加载 user_taggedartists.dat 信息...")
try:
    # 重点修改：指定编码为 'latin1'
    user_tagged_artists_df = pd.read_csv(USER_TAGGED_ARTISTS_DAT_PATH, sep='\t', encoding='latin1')
    print(f"user_tagged_artists.dat 加载完成，共有 {len(user_tagged_artists_df)} 条记录。")
except FileNotFoundError:
    print(f"错误: 未找到 {USER_TAGGED_ARTISTS_DAT_PATH}。请检查文件路径。")
    exit()
except UnicodeDecodeError as e:
    print(f"解码错误: 无法使用 latin1 编码读取 {USER_TAGGED_ARTISTS_DAT_PATH}。尝试其他编码或检查文件。错误信息: {e}")
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

# 新增：预处理标签数据
print("\n预处理标签数据...")
# 尝试从缓存加载原始 artist_to_tags 和 unique_tags
artist_to_tags_raw = None
unique_tags_raw = None

if os.path.exists(ARTIST_TO_TAGS_CACHE) and os.path.exists(UNIQUE_TAGS_CACHE):
    print(f"尝试从缓存加载原始艺术家-标签映射和唯一标签集合...")
    try:
        with open(ARTIST_TO_TAGS_CACHE, 'rb') as f:
            artist_to_tags_raw = pickle.load(f)
        with open(UNIQUE_TAGS_CACHE, 'rb') as f:
            unique_tags_raw = pickle.load(f)
        print("原始艺术家-标签映射和唯一标签集合从缓存加载成功。")
    except Exception as e:
        print(f"加载原始缓存失败: {e}。将重新处理标签数据。")
        artist_to_tags_raw = None
        unique_tags_raw = None

if artist_to_tags_raw is None or unique_tags_raw is None:
    start_time_tag_process = time.time()
    artist_to_tags_raw, _, unique_tags_raw = UserMatrix2.preprocess_tag_data(
        user_tagged_artists_df,
        artist_id_to_idx_train,  # 使用训练集中的艺术家ID映射
        tag_id_to_value
    )
    with open(ARTIST_TO_TAGS_CACHE, 'wb') as f:
        pickle.dump(artist_to_tags_raw, f)
    with open(UNIQUE_TAGS_CACHE, 'wb') as f:
        pickle.dump(unique_tags_raw, f)
    end_time_tag_process = time.time()
    print(f"标签数据预处理并保存到原始缓存完成，耗时: {end_time_tag_process - start_time_tag_process:.2f} 秒。")

print(f"共加载 {len(unique_tags_raw)} 个唯一标签 (原始)。")

# --- 新增：标签频率过滤 ---
print("\n进行标签频率过滤...")
artist_to_tags_filtered = {}
unique_tags_filtered = set()

# 尝试从缓存加载过滤后的标签数据
# 确保缓存文件名的区分度，可以加上 MIN_TAG_FREQ_GLOBAL 和 MAX_TAG_FREQ_RATIO_GLOBAL
FILTERED_ARTIST_TO_TAGS_CACHE_SPECIFIC = os.path.join(CACHE_DIR,
                                                      f'filtered_artist_to_tags_min{MIN_TAG_FREQ_GLOBAL}_max{int(MAX_TAG_FREQ_RATIO_GLOBAL * 100)}.pkl')
FILTERED_UNIQUE_TAGS_CACHE_SPECIFIC = os.path.join(CACHE_DIR,
                                                   f'filtered_unique_tags_min{MIN_TAG_FREQ_GLOBAL}_max{int(MAX_TAG_FREQ_RATIO_GLOBAL * 100)}.pkl')

if os.path.exists(FILTERED_ARTIST_TO_TAGS_CACHE_SPECIFIC) and os.path.exists(FILTERED_UNIQUE_TAGS_CACHE_SPECIFIC):
    print(f"尝试从缓存加载过滤后的艺术家-标签映射和唯一标签集合 (根据当前过滤参数)...")
    try:
        with open(FILTERED_ARTIST_TO_TAGS_CACHE_SPECIFIC, 'rb') as f:
            artist_to_tags_filtered = pickle.load(f)
        with open(FILTERED_UNIQUE_TAGS_CACHE_SPECIFIC, 'rb') as f:
            unique_tags_filtered = pickle.load(f)
        print("过滤后的标签数据从缓存加载成功。")
    except Exception as e:
        print(f"加载过滤缓存失败: {e}。将重新过滤标签数据。")
        artist_to_tags_filtered = {}
        unique_tags_filtered = set()

if not artist_to_tags_filtered or not unique_tags_filtered:
    start_time_filter = time.time()
    # 1. 统计每个标签在多少个艺术家中出现
    tag_document_counts = defaultdict(int)
    for artist_id, tags in artist_to_tags_raw.items():
        for tag in tags:
            tag_document_counts[tag] += 1

    total_artists_in_raw_data = len(artist_to_tags_raw)

    # 确定要保留的标签
    valid_tags_set = set()
    for tag, count in tag_document_counts.items():
        if count >= MIN_TAG_FREQ_GLOBAL and \
                count / total_artists_in_raw_data <= MAX_TAG_FREQ_RATIO_GLOBAL:
            valid_tags_set.add(tag)

    # 2. 过滤 artist_to_tags
    for artist_id, tags in artist_to_tags_raw.items():
        filtered_tags_for_artist = [tag for tag in tags if tag in valid_tags_set]
        if filtered_tags_for_artist:  # 只有当艺术家至少有一个有效标签时才保留
            artist_to_tags_filtered[artist_id] = filtered_tags_for_artist
            unique_tags_filtered.update(filtered_tags_for_artist)

    # 保存过滤后的数据到缓存
    with open(FILTERED_ARTIST_TO_TAGS_CACHE_SPECIFIC, 'wb') as f:
        pickle.dump(artist_to_tags_filtered, f)
    with open(FILTERED_UNIQUE_TAGS_CACHE_SPECIFIC, 'wb') as f:
        pickle.dump(unique_tags_filtered, f)

    end_time_filter = time.time()
    print(f"标签频率过滤完成，耗时: {end_time_filter - start_time_filter:.2f} 秒。")

print(f"过滤后，共保留 {len(unique_tags_filtered)} 个唯一标签。")
print(f"过滤后，共保留 {len(artist_to_tags_filtered)} 位艺术家的标签信息。")

# 将过滤后的标签数据用于后续的 Content-Based 推荐
artist_to_tags = artist_to_tags_filtered
unique_tags = unique_tags_filtered

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

# --- 阶段 3: 计算协同过滤相似度 (基于训练矩阵) ---
print("\n--- 阶段 3: 计算协同过滤相似度 ---")

# 计算并缓存融合用户相似度矩阵 (User-Based CF with Social Fusion)
user_similarity_matrix_fused = None
fused_sim_cache_path = os.path.join(CACHE_DIR, f'user_similarity_fused_alpha_{BEST_ALPHA_FUSION_UB}.pkl')

if os.path.exists(fused_sim_cache_path):
    print(f"尝试从缓存文件 '{fused_sim_cache_path}' 加载融合用户相似度矩阵...")
    try:
        user_similarity_matrix_fused = joblib.load(fused_sim_cache_path)
        if user_similarity_matrix_fused.shape[0] != user_artist_matrix_train.shape[0] or \
                user_similarity_matrix_fused.shape[1] != user_artist_matrix_train.shape[0]:
            print("警告：缓存的融合相似度矩阵形状不匹配，将重新计算。")
            user_similarity_matrix_fused = None
    except Exception as e:
        print(f"加载缓存失败: {e}。将重新计算融合用户相似度矩阵。")
        user_similarity_matrix_fused = None

if user_similarity_matrix_fused is None:
    start_time_ub_sim = time.time()
    print(f"\n计算 User-Based CF 融合用户相似度矩阵 (Alpha={BEST_ALPHA_FUSION_UB})...")
    user_similarity_matrix_fused = ra.calculate_user_similarity_social_fused(
        user_artist_matrix_train,
        user_friends_data,
        user_id_to_idx_train,
        idx_to_user_id_train,
        alpha=BEST_ALPHA_FUSION_UB
    )
    joblib.dump(user_similarity_matrix_fused, fused_sim_cache_path)
    end_time_ub_sim = time.time()
    print(f"融合用户相似度矩阵计算并保存完成，耗时: {end_time_ub_sim - start_time_ub_sim:.2f} 秒。")

print("融合用户相似度矩阵维度:", user_similarity_matrix_fused.shape)

# 计算并缓存物品相似度矩阵 (Item-Based CF)
item_similarity_matrix = None

if os.path.exists(ITEM_SIMILARITY_CACHE):
    print(f"\n尝试从缓存文件 '{ITEM_SIMILARITY_CACHE}' 加载物品相似度矩阵...")
    try:
        item_similarity_matrix = joblib.load(ITEM_SIMILARITY_CACHE)
        if item_similarity_matrix.shape[0] != user_artist_matrix_train.shape[1] or \
                item_similarity_matrix.shape[1] != user_artist_matrix_train.shape[1]:
            print("警告：缓存的物品相似度矩阵形状不匹配，将重新计算。")
            item_similarity_matrix = None
    except Exception as e:
        print(f"加载缓存失败: {e}。将重新计算物品相似度矩阵。")
        item_similarity_matrix = None

if item_similarity_matrix is None:
    start_time_ib_sim = time.time()
    print("\n计算物品（艺术家）相似度...")
    artist_user_matrix_train = user_artist_matrix_train.T  # CSR matrix 的转置是 CSC matrix
    item_similarity_matrix = ra.calculate_item_similarity_cosine(artist_user_matrix_train)
    joblib.dump(item_similarity_matrix, ITEM_SIMILARITY_CACHE)
    end_time_ib_sim = time.time()
    print(
        f"物品相似度矩阵已计算并保存到缓存文件 '{ITEM_SIMILARITY_CACHE}'，耗时: {end_time_ib_sim - start_time_ib_sim:.2f} 秒。")

print("物品相似度矩阵维度:", item_similarity_matrix.shape)

# --- 阶段 4: 推荐生成和评估 ---
print("\n--- 阶段 4: 推荐生成和评估 ---")

# 获取训练集和测试集中都存在的用户 (用于评估的用户列表)
users_in_train_and_test = set(user_id_to_idx_train.keys()).intersection(user_item_ratings_test.keys())
users_to_evaluate_ids_raw = list(users_in_train_and_test)

users_for_eval = []
if FAST_EVAL_MODE:
    np.random.seed(42)  # 保证每次运行选到的用户相同
    users_for_eval = np.random.choice(users_to_evaluate_ids_raw,
                                      min(FAST_EVAL_USER_LIMIT, len(users_to_evaluate_ids_raw)),
                                      replace=False).tolist()
    print(f"!!! 快速评估模式：只评估 {len(users_for_eval)} 个用户 !!!")
else:
    users_for_eval = users_to_evaluate_ids_raw
    print(f"--- 完整评估模式：评估所有 {len(users_for_eval)} 个符合条件的用户 ---")

# 进一步过滤测试集，只保留那些在训练矩阵中存在的用户和艺术家
eval_test_ratings_filtered = {}
for user_id in users_for_eval:
    current_user_test_artists = user_item_ratings_test.get(user_id, {})
    filtered_artists = {
        artist_id: count for artist_id, count in current_user_test_artists.items()
        if artist_id in artist_id_to_idx_train  # 艺术家在训练集构建的矩阵中存在
    }
    if filtered_artists:  # 如果过滤后用户仍有有效测试记录
        eval_test_ratings_filtered[user_id] = filtered_artists

print(f"最终用于评估的用户数量: {len(eval_test_ratings_filtered)}")

# --- 评估 User-Based CF (Social Fused) 使用最佳参数 ---
if user_similarity_matrix_fused is not None and eval_test_ratings_filtered:
    print(f"\n--- 评估 User-Based CF (Social Fused) ---")
    start_time_ub_eval = time.time()
    avg_precision_ub, avg_recall_ub = re.evaluate_model(
        recommendation_function=lambda uid: [
            rec[0] for rec in ra.recommend_user_based_cf(
                user_id=uid,
                user_artist_matrix=user_artist_matrix_train,
                user_similarity_matrix=user_similarity_matrix_fused,
                user_id_to_idx=user_id_to_idx_train,
                idx_to_user_id=idx_to_user_id_train,
                user_item_ratings_train=user_item_ratings_train,
                num_recommendations=NUM_RECOMMENDATIONS,
                k_neighbors=BEST_K_NEIGHBORS_UB
            )
        ],
        user_item_ratings_test=eval_test_ratings_filtered,
        users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),
        top_k=NUM_RECOMMENDATIONS
    )
    end_time_ub_eval = time.time()
    # 计算 F1-score
    f1_score_ub = 0.0
    if (avg_precision_ub + avg_recall_ub) > 0:
        f1_score_ub = 2 * (avg_precision_ub * avg_recall_ub) / \
                      (avg_precision_ub + avg_recall_ub)
    print(f"User-Based CF (Social Fused) Precision@{NUM_RECOMMENDATIONS}: {avg_precision_ub:.4f}")
    print(f"User-Based CF (Social Fused) Recall@{NUM_RECOMMENDATIONS}: {avg_recall_ub:.4f}")
    print(f"User-Based CF (Social Fused) F1-score@{NUM_RECOMMENDATIONS}: {f1_score_ub:.4f}")
    print(f"User-Based CF (Social Fused) 评估耗时: {end_time_ub_eval - start_time_ub_eval:.2f} 秒。")
else:
    print("没有可用于评估 User-Based CF 的有效测试用户或相似度矩阵为空。")

# --- 评估 Item-Based CF 使用最佳参数 ---
if item_similarity_matrix is not None and eval_test_ratings_filtered:
    print(f"\n--- 评估 Item-Based CF ---")
    start_time_ib_eval = time.time()
    avg_precision_ib, avg_recall_ib = re.evaluate_model(
        recommendation_function=lambda uid: [
            rec[0] for rec in ra.recommend_item_based_cf(
                user_id=uid,
                user_artist_matrix=user_artist_matrix_train,
                item_similarity_matrix=item_similarity_matrix,
                user_id_to_idx=user_id_to_idx_train,
                artist_id_to_idx=artist_id_to_idx_train,
                idx_to_artist_id=idx_to_artist_id_train,
                user_item_ratings_train=user_item_ratings_train,
                num_recommendations=NUM_RECOMMENDATIONS,
                k_neighbors=BEST_K_NEIGHBORS_IB
            )
        ],
        user_item_ratings_test=eval_test_ratings_filtered,
        users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),
        top_k=NUM_RECOMMENDATIONS
    )
    end_time_ib_eval = time.time()
    # 计算 F1-score
    f1_score_ib = 0.0
    if (avg_precision_ib + avg_recall_ib) > 0:
        f1_score_ib = 2 * (avg_precision_ib * avg_recall_ib) / \
                      (avg_precision_ib + avg_recall_ib)
    print(f"Item-Based CF Precision@{NUM_RECOMMENDATIONS}: {avg_precision_ib:.4f}")
    print(f"Item-Based CF Recall@{NUM_RECOMMENDATIONS}: {avg_recall_ib:.4f}")
    print(f"Item-Based CF F1-score@{NUM_RECOMMENDATIONS}: {f1_score_ib:.4f}")
    print(f"Item-Based CF 评估耗时: {end_time_ib_eval - start_time_ib_eval:.2f} 秒。")
else:
    print("没有可用于评估 Item-Based CF 的有效测试用户或物品相似度矩阵为空。")

# --- 评估 Content-Based 推荐 ---
if eval_test_ratings_filtered and artist_to_tags:  # 确保有可评估的用户和标签数据
    print(f"\n--- 评估 Content-Based 推荐 (Min Tags Per Artist={BEST_MIN_TAGS_PER_ARTIST_CB}) ---")
    start_time_cb_eval = time.time()
    avg_precision_cb, avg_recall_cb = re.evaluate_model(
        recommendation_function=lambda uid: [
            rec[0] for rec in ra.recommend_content_based(
                user_id=uid,
                user_item_ratings_train=user_item_ratings_train,
                artist_to_tags=artist_to_tags,  # 使用过滤后的标签数据
                unique_tags=unique_tags,  # 使用过滤后的标签数据
                artist_id_to_idx=artist_id_to_idx_train,
                idx_to_artist_id=idx_to_artist_id_train,
                num_recommendations=NUM_RECOMMENDATIONS,
                min_tags_per_artist=BEST_MIN_TAGS_PER_ARTIST_CB  # 传递参数
            )
        ],
        user_item_ratings_test=eval_test_ratings_filtered,
        users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),
        top_k=NUM_RECOMMENDATIONS
    )
    end_time_cb_eval = time.time()
    # 计算 F1-score
    f1_score_cb = 0.0
    if (avg_precision_cb + avg_recall_cb) > 0:
        f1_score_cb = 2 * (avg_precision_cb * avg_recall_cb) / \
                      (avg_precision_cb + avg_recall_cb)
    print(f"Content-Based Precision@{NUM_RECOMMENDATIONS}: {avg_precision_cb:.4f}")
    print(f"Content-Based Recall@{NUM_RECOMMENDATIONS}: {avg_recall_cb:.4f}")
    print(f"Content-Based F1-score@{NUM_RECOMMENDATIONS}: {f1_score_cb:.4f}")
    print(f"Content-Based 评估耗时: {end_time_cb_eval - start_time_cb_eval:.2f} 秒。")
else:
    print("没有可用于评估 Content-Based 推荐的有效测试用户或标签数据为空。")

# --- 评估 Hybrid Weighted 推荐 (使用最佳参数) ---
# 仅当所有必要数据都可用时才进行评估
if eval_test_ratings_filtered and user_similarity_matrix_fused is not None and artist_to_tags:
    print(
        f"\n--- 评估 Hybrid Weighted 推荐 (CB Weight={BEST_CB_WEIGHT_HYBRID:.2f}, Min Tags Per Artist={BEST_MIN_TAGS_PER_ARTIST_CB}) ---")
    start_time_hybrid_eval = time.time()
    avg_precision_hybrid, avg_recall_hybrid = re.evaluate_model(
        recommendation_function=lambda uid: ra.recommend_hybrid_weighted(
            user_id=uid,
            user_artist_matrix_train=user_artist_matrix_train,
            user_similarity_matrix_fused=user_similarity_matrix_fused,
            user_id_to_idx_train=user_id_to_idx_train,
            idx_to_user_id_train=idx_to_user_id_train,
            artist_id_to_idx_train=artist_id_to_idx_train,
            idx_to_artist_id_train=idx_to_artist_id_train,
            user_item_ratings_train=user_item_ratings_train,
            artist_to_tags=artist_to_tags,  # 使用过滤后的标签数据
            unique_tags=unique_tags,  # 使用过滤后的标签数据
            num_recommendations=NUM_RECOMMENDATIONS,
            ub_k_neighbors=BEST_K_NEIGHBORS_UB,
            cb_weight=BEST_CB_WEIGHT_HYBRID,  # 使用最佳权重
            min_tags_per_artist=BEST_MIN_TAGS_PER_ARTIST_CB  # 使用最佳最小标签数
        ),
        user_item_ratings_test=eval_test_ratings_filtered,
        users_to_evaluate_ids=list(eval_test_ratings_filtered.keys()),
        top_k=NUM_RECOMMENDATIONS
    )
    end_time_hybrid_eval = time.time()

    # 计算 F1-score
    f1_score_hybrid = 0.0
    if (avg_precision_hybrid + avg_recall_hybrid) > 0:
        f1_score_hybrid = 2 * (avg_precision_hybrid * avg_recall_hybrid) / \
                          (avg_precision_hybrid + avg_recall_hybrid)

    print(f"Hybrid Weighted Precision@{NUM_RECOMMENDATIONS}: {avg_precision_hybrid:.4f}")
    print(f"Hybrid Weighted Recall@{NUM_RECOMMENDATIONS}: {avg_recall_hybrid:.4f}")
    print(f"Hybrid Weighted F1-score@{NUM_RECOMMENDATIONS}: {f1_score_hybrid:.4f}")
    print(f"Hybrid Weighted 评估耗时: {end_time_hybrid_eval - start_time_hybrid_eval:.2f} 秒。")

else:
    print("没有可用于评估 Hybrid Weighted 推荐的有效测试用户、相似度矩阵或标签数据为空。")

print("\n--- 音乐推荐系统关闭 ---")
