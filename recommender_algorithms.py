# recommender_algorithms.py

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict  # 引入 defaultdict 用于方便地构建用户标签偏好
import math  # 引入 math 用于计算 log


# --- 现有函数 ---

def calculate_user_similarity_cosine(user_artist_matrix):
    """
    计算用户之间的余弦相似度。
    Args:
        user_artist_matrix (csr_matrix): 用户-艺术家交互矩阵。
    Returns:
        np.array: 稠密的用户相似度矩阵。
    """
    if not isinstance(user_artist_matrix, csr_matrix):
        print("警告: calculate_user_similarity_cosine 接收到的不是 csr_matrix，正在尝试转换。")
        user_artist_matrix = csr_matrix(user_artist_matrix)

    return cosine_similarity(user_artist_matrix)


def calculate_item_similarity_cosine(artist_user_matrix):
    """
    计算物品（艺术家）之间的余弦相似度。
    Args:
        artist_user_matrix (csr_matrix): 艺术家-用户交互矩阵（用户-艺术家矩阵的转置）。
    Returns:
        np.array: 稠密的物品相似度矩阵。
    """
    if not isinstance(artist_user_matrix, csr_matrix):
        print("警告: calculate_item_similarity_cosine 接收到的不是 csr_matrix，正在尝试转换。")
        artist_user_matrix = csr_matrix(artist_user_matrix)

    return cosine_similarity(artist_user_matrix)


def calculate_user_similarity_social_fused(user_artist_matrix, user_friends_data,
                                           user_id_to_idx, idx_to_user_id, alpha=0.5):
    """
    通过融合行为（余弦）和社交相似度来计算用户相似度。
    Args:
        user_artist_matrix (csr_matrix): 用户-艺术家交互矩阵。
        user_friends_data (dict): 用户好友关系的字典，键是用户 ID，值是好友 ID 的集合。
        user_id_to_idx (dict): 从实际用户 ID 到矩阵索引的映射。
        idx_to_user_id (dict): 从矩阵索引到实际用户 ID 的映射。
        alpha (float): 社交相似度的加权因子 (0 到 1)。
                       alpha=0 意味着纯行为相似度，alpha=1 意味着纯社交相似度。
    Returns:
        np.array: 融合后的用户相似度稠密矩阵。
    """
    num_users = user_artist_matrix.shape[0]
    fused_similarity_matrix = np.zeros((num_users, num_users))

    # 计算行为相似度（余弦相似度）
    behavioral_similarity_matrix = cosine_similarity(user_artist_matrix)

    # 融合社交相似度
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                fused_similarity_matrix[i, j] = 1.0  # 用户与自己的相似度为 1
                continue

            user_id_i = idx_to_user_id.get(i)
            user_id_j = idx_to_user_id.get(j)

            # 如果任何一个用户ID不在映射中，或者根本没在训练集中出现，则跳过或仅用行为相似度
            if user_id_i is None or user_id_j is None:
                fused_similarity_matrix[i, j] = behavioral_similarity_matrix[i, j]
                continue

            bh_sim = behavioral_similarity_matrix[i, j]

            # 判断是否是好友：在 user_friends_data 中查找
            # 注意：user_friends_data 可能只包含部分用户（那些有好友的用户）
            # 所以要用 .get(user_id_i, set()) 来安全获取
            is_friend = 1.0 if user_id_j in user_friends_data.get(user_id_i, set()) else 0.0

            # 融合相似度
            fused_similarity_matrix[i, j] = (1 - alpha) * bh_sim + alpha * is_friend

    return fused_similarity_matrix


def recommend_user_based_cf(user_id, user_artist_matrix, user_similarity_matrix,
                            user_id_to_idx, idx_to_user_id, user_item_ratings_train,
                            num_recommendations=10, k_neighbors=50):
    """
    基于用户的协同过滤推荐函数。
    参数:
        user_id (int): 要为其生成推荐的用户 ID。
        user_artist_matrix (csr_matrix): 用户-艺术家交互矩阵（主要用于获取维度信息，实际推荐基于相似度）。
        user_similarity_matrix (np.array): 用户相似度矩阵（可以是融合后的）。
        user_id_to_idx (dict): 从实际用户 ID 到矩阵索引的映射。
        idx_to_user_id (dict): 从矩阵索引到实际用户 ID 的映射。
        user_item_ratings_train (dict): 训练集中用户-艺术家播放次数的字典。
        num_recommendations (int): 要推荐的艺术家数量。
        k_neighbors (int): 邻居的数量。
    返回:
        list: 推荐的艺术家 ID 和对应分数 (artist_id, score) 元组的列表。
    """
    if user_id not in user_id_to_idx:
        return []  # 如果用户不在训练集中，无法推荐

    target_user_idx = user_id_to_idx[user_id]

    # 获取目标用户的相似度（与其他所有用户）
    user_similarities = user_similarity_matrix[target_user_idx]

    # 找到 K 个最相似的邻居，排除目标用户本身
    similar_users_indices = np.argsort(user_similarities)[::-1]

    valid_neighbors = []
    for idx in similar_users_indices:
        if idx == target_user_idx:  # 排除目标用户自己
            continue
        # 确保邻居在 user_id_to_idx 映射中存在（即在训练数据中是有效用户）
        if idx_to_user_id.get(idx) is not None:
            valid_neighbors.append(idx)
        if len(valid_neighbors) >= k_neighbors:
            break

    if not valid_neighbors:
        return []  # 如果没有找到邻居，则无法推荐

    artist_scores = {}

    # 获取目标用户已经听过的艺术家，避免重复推荐
    listened_artists = set(user_item_ratings_train.get(user_id, {}).keys())

    # 遍历每个邻居
    for neighbor_idx in valid_neighbors:
        neighbor_user_id = idx_to_user_id[neighbor_idx]  # 获取实际用户ID
        similarity = user_similarities[neighbor_idx]

        # 获取邻居听过的艺术家及其播放次数
        neighbor_listened_artists = user_item_ratings_train.get(neighbor_user_id, {})
        for artist_id, listen_count in neighbor_listened_artists.items():
            if artist_id not in listened_artists:  # 只考虑目标用户未听过的艺术家
                # 预测评分：相似度 * 邻居的播放次数
                artist_scores.setdefault(artist_id, 0.0)
                artist_scores[artist_id] += similarity * listen_count

    # 按预测评分排序
    recommended_artists_with_scores = sorted(artist_scores.items(), key=lambda item: item[1], reverse=True)
    # 返回包含分数和艺术家ID的列表
    return recommended_artists_with_scores[:num_recommendations]


def recommend_item_based_cf(user_id, user_artist_matrix, item_similarity_matrix,
                            user_id_to_idx, artist_id_to_idx, idx_to_artist_id, user_item_ratings_train,
                            num_recommendations=10, k_neighbors=50):
    """
    基于物品（艺术家）的协同过滤推荐函数。
    参数:
        user_id (int): 要为其生成推荐的用户 ID。
        user_artist_matrix (csr_matrix): 用户-艺术家交互矩阵（主要用于获取维度信息）。
        item_similarity_matrix (np.array): 物品相似度矩阵。
        user_id_to_idx (dict): 从实际用户 ID 到矩阵索引的映射。
        artist_id_to_idx (dict): 从实际艺术家 ID 到矩阵索引的映射。
        idx_to_artist_id (dict): 从矩阵索引到实际艺术家 ID 的映射。
        user_item_ratings_train (dict): 训练集中用户-艺术家播放次数的字典。
        num_recommendations (int): 要推荐的艺术家数量。
        k_neighbors (int): 邻居的数量。
    返回:
        list: 推荐的艺术家 ID 和对应分数 (artist_id, score) 元组的列表。
    """
    if user_id not in user_id_to_idx:
        return []

    target_user_idx = user_id_to_idx[user_id]

    # 获取目标用户已经听过的艺术家及其播放次数
    user_listened_artists = user_item_ratings_train.get(user_id, {})

    # 存储未听过的艺术家的预测评分
    artist_scores = {}

    for listened_artist_id, listen_count in user_listened_artists.items():
        if listened_artist_id not in artist_id_to_idx:
            continue  # 如果听过的艺术家不在矩阵中（可能因为过滤掉了），跳过

        listened_artist_idx = artist_id_to_idx[listened_artist_id]

        # 获取与已听艺术家相似的艺术家（邻居）
        item_similarities = item_similarity_matrix[listened_artist_idx]

        # 找到 K 个最相似的艺术家，排除自己
        similar_artist_indices = np.argsort(item_similarities)[::-1]

        valid_neighbors = []
        for idx in similar_artist_indices:
            if idx == listened_artist_idx:  # 排除物品本身
                continue
            if idx_to_artist_id.get(idx) is not None:  # 确保邻居艺术家在映射中
                valid_neighbors.append(idx)
            if len(valid_neighbors) >= k_neighbors:
                break

        for neighbor_artist_idx in valid_neighbors:
            neighbor_artist_id = idx_to_artist_id[neighbor_artist_idx]

            # 如果目标用户已经听过这个邻居艺术家，则跳过
            if neighbor_artist_id in user_listened_artists:
                continue

            similarity = item_similarities[neighbor_artist_idx]

            # 预测评分：相似度 * 目标用户对已听艺术家的播放次数
            artist_scores.setdefault(neighbor_artist_id, 0.0)
            artist_scores[neighbor_artist_id] += similarity * listen_count

    # 按预测评分排序
    recommended_artists_with_scores = sorted(artist_scores.items(), key=lambda item: item[1], reverse=True)
    # 返回包含分数和艺术家ID的列表
    return recommended_artists_with_scores[:num_recommendations]


def calculate_jaccard_similarity(set1, set2):
    """
    计算两个集合之间的Jaccard相似度。
    Args:
        set1 (set): 集合1。
        set2 (set): 集合2。
    Returns:
        float: Jaccard相似度。如果两个集合都为空，返回0。
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0  # 避免除以零
    return intersection / union


def calculate_idf(artist_to_tags, unique_tags, total_artists_with_tags):
    """
    计算每个标签的逆文档频率 (IDF)。
    Args:
        artist_to_tags (dict): 艺术家ID到其关联标签列表的映射。
        unique_tags (set): 所有唯一的标签值集合。
        total_artists_with_tags (int): 拥有标签的艺术家总数。
    Returns:
        dict: 标签到其IDF值的映射。
    """
    idf_scores = {}

    # 统计每个标签在多少个艺术家中出现
    tag_document_counts = defaultdict(int)
    for artist_id, tags in artist_to_tags.items():
        for tag in tags:
            tag_document_counts[tag] += 1

    # 计算 IDF
    for tag in unique_tags:
        # 避免除以零，并使用平滑项 +1
        idf_scores[tag] = math.log10(total_artists_with_tags / (tag_document_counts[tag] + 1)) + 1
    return idf_scores


def recommend_content_based(user_id, user_item_ratings_train, artist_to_tags,
                            unique_tags, artist_id_to_idx, idx_to_artist_id,
                            num_recommendations=10, min_tags_per_artist=2):  # 新增参数
    """
    基于内容的推荐函数（使用 TF-IDF）。
    根据用户已听艺术家的标签偏好来推荐新艺术家。

    Args:
        user_id (int): 要为其生成推荐的用户 ID。
        user_item_ratings_train (dict): 训练集中用户-艺术家播放次数的字典。
        artist_to_tags (dict): 艺术家ID到其关联标签列表的映射。
        unique_tags (set): 所有唯一的标签值集合。
        artist_id_to_idx (dict): 从实际艺术家 ID 到矩阵索引的映射 (用于过滤有效艺术家)。
        idx_to_artist_id (dict): 从矩阵索引到实际艺术家 ID 的映射 (用于过滤有效艺术家)。
        num_recommendations (int): 要推荐的艺术家数量。
        min_tags_per_artist (int): 艺术家被考虑的最低标签数量。

    Returns:
        list: 推荐的艺术家 ID 和对应分数 (artist_id, score) 元组的列表。
    """
    if user_id not in user_item_ratings_train:
        return []  # 如果用户没有历史记录，无法进行内容基推荐

    user_listened_artists_with_ratings = user_item_ratings_train.get(user_id, {})
    listened_artist_ids = set(user_listened_artists_with_ratings.keys())

    # 计算标签的 IDF 值
    # 仅考虑那些至少有 min_tags_per_artist 个标签的艺术家来计算 IDF
    filtered_artists_for_idf = {
        art_id: tags for art_id, tags in artist_to_tags.items()
        if len(tags) >= min_tags_per_artist
    }
    total_artists_with_sufficient_tags = len(filtered_artists_for_idf)

    if total_artists_with_sufficient_tags == 0:
        return []  # 如果没有艺术家有足够标签，无法进行内容基推荐

    idf_scores = calculate_idf(filtered_artists_for_idf, unique_tags, total_artists_with_sufficient_tags)

    # 1. 构建艺术家标签的 TF-IDF 向量 (这里简化为字典形式)
    artist_tag_tfidf_vectors = {}
    for artist_id, tags in artist_to_tags.items():
        total_tags_for_artist = len(tags)  # 获取该艺术家的总标签数

        # 过滤掉标签数量不足的艺术家
        if total_tags_for_artist < min_tags_per_artist:
            continue

        artist_tf = defaultdict(float)
        # 计算 TF (这里是标签频率)
        for tag in tags:
            artist_tf[tag] += 1  # 原始计数

        # 计算 TF-IDF
        artist_tfidf_vector = {}
        for tag, tf_count in artist_tf.items():
            # TF = 标签出现次数 / 艺术家总标签数 (标准化词频)
            tf = tf_count / total_tags_for_artist
            artist_tfidf_vector[tag] = tf * idf_scores.get(tag, 0)  # 使用 get 避免标签不存在IDF
        artist_tag_tfidf_vectors[artist_id] = artist_tfidf_vector

    # 2. 构建用户标签偏好档案 (User-Tag Profile) - 平均化 TF-IDF 累加
    user_tag_profile_tfidf = defaultdict(float)
    count_artists_in_profile = 0  # 统计实际用于构建用户画像的艺术家数量

    for artist_id, _ in user_listened_artists_with_ratings.items():  # 移除 listen_count 的直接使用
        if artist_id in artist_tag_tfidf_vectors:  # 确保艺术家有 TF-IDF 向量 (即通过了标签数量过滤)
            artist_tfidf_vector = artist_tag_tfidf_vectors[artist_id]
            # 累加艺术家标签的 TF-IDF 分数
            for tag, tfidf_score in artist_tfidf_vector.items():
                user_tag_profile_tfidf[tag] += tfidf_score
            count_artists_in_profile += 1  # 统计实际用于构建画像的艺术家数量

    if not user_tag_profile_tfidf or count_artists_in_profile == 0:  # 如果用户听过的艺术家都没有TF-IDF标签信息，或者没有有效艺术家，无法进行内容基推荐
        return []

    # 平均化用户偏好档案
    for tag in user_tag_profile_tfidf:
        user_tag_profile_tfidf[tag] /= count_artists_in_profile

    # 将用户偏好档案转换为向量形式，以便计算余弦相似度
    user_profile_vec = np.zeros(len(unique_tags))
    tag_to_idx = {tag: i for i, tag in enumerate(sorted(list(unique_tags)))}  # 为标签创建索引

    for tag, score in user_tag_profile_tfidf.items():
        if tag in tag_to_idx:
            user_profile_vec[tag_to_idx[tag]] = score

    # 归一化用户偏好向量，如果需要 (余弦相似度通常不需要额外归一化输入向量，因为它内部会归一化)
    # user_profile_vec_norm = np.linalg.norm(user_profile_vec)
    # if user_profile_vec_norm > 0:
    #     user_profile_vec = user_profile_vec / user_profile_vec_norm

    # 3. 预测未听过艺术家的评分
    candidate_artist_scores = {}

    # 遍历所有在训练集中的有效艺术家
    for artist_id in artist_id_to_idx.keys():
        if artist_id in listened_artist_ids:
            continue  # 跳过用户已经听过的艺术家

        # 确保候选艺术家有足够标签，且其TF-IDF向量已构建
        if artist_id not in artist_tag_tfidf_vectors:
            continue

        artist_tfidf_vector_dict = artist_tag_tfidf_vectors[artist_id]

        # 将艺术家标签TF-IDF向量转换为与用户偏好向量相同维度的向量，以便计算余弦相似度
        artist_vec = np.zeros(len(unique_tags))
        for tag, tfidf_score in artist_tfidf_vector_dict.items():
            if tag in tag_to_idx:
                artist_vec[tag_to_idx[tag]] = tfidf_score

        # 计算用户偏好向量与艺术家向量之间的余弦相似度
        # reshape(-1, 1) or (1, -1) for single sample if using sklearn.metrics.pairwise.cosine_similarity
        # Here, we do manual dot product and norm for simplicity

        # 计算点积
        dot_product = np.dot(user_profile_vec, artist_vec)

        # 计算向量模长
        user_norm = np.linalg.norm(user_profile_vec)
        artist_norm = np.linalg.norm(artist_vec)

        score = 0.0
        if user_norm > 0 and artist_norm > 0:
            score = dot_product / (user_norm * artist_norm)

        if score > 0:  # 只保留有正向相似度的艺术家
            candidate_artist_scores[artist_id] = score

    # 4. 排序并返回 Top-N 推荐
    recommended_artists_with_scores = sorted(
        candidate_artist_scores.items(), key=lambda item: item[1], reverse=True
    )

    return recommended_artists_with_scores[:num_recommendations]


def recommend_hybrid_weighted(user_id,
                              user_artist_matrix_train, user_similarity_matrix_fused,
                              user_id_to_idx_train, idx_to_user_id_train, artist_id_to_idx_train,
                              idx_to_artist_id_train,
                              user_item_ratings_train,
                              artist_to_tags, unique_tags,  # 标签数据参数
                              num_recommendations=10,
                              ub_k_neighbors=200,
                              cb_weight=0.5,
                              min_tags_per_artist=2):  # 新增参数
    """
    加权混合推荐函数，融合 User-Based CF (Social Fused) 和 Content-Based 推荐。

    Args:
        user_id (int): 要为其生成推荐的用户 ID。
        user_artist_matrix_train (csr_matrix): 用户-艺术家训练矩阵。
        user_similarity_matrix_fused (np.array): 融合后的用户相似度矩阵。
        user_id_to_idx_train (dict): 用户ID到矩阵索引的映射。
        idx_to_user_id_train (dict): 矩阵索引到用户ID的映射。
        artist_id_to_idx_train (dict): 艺术家ID到矩阵索引的映射。
        idx_to_artist_id_train (dict): 矩阵索引到艺术家ID的映射。
        user_item_ratings_train (dict): 训练集中用户-艺术家播放次数的字典。
        artist_to_tags (dict): 艺术家ID到其关联标签列表的映射。
        unique_tags (set): 所有唯一的标签值集合。
        num_recommendations (int): 要推荐的艺术家数量。
        ub_k_neighbors (int): User-Based CF 的邻居数量。
        cb_weight (float): Content-Based 推荐结果的权重 (0到1)。
                           (1 - cb_weight) 将是 User-Based CF 的权重。
        min_tags_per_artist (int): 艺术家被考虑的最低标签数量（用于Content-Based）。

    Returns:
        list: 推荐的艺术家 ID 列表。
    """

    # 获取 User-Based CF 的推荐分数
    ub_recs_with_scores = recommend_user_based_cf(
        user_id=user_id,
        user_artist_matrix=user_artist_matrix_train,
        user_similarity_matrix=user_similarity_matrix_fused,
        user_id_to_idx=user_id_to_idx_train,
        idx_to_user_id=idx_to_user_id_train,
        user_item_ratings_train=user_item_ratings_train,
        num_recommendations=num_recommendations * 2,  # 临时获取更多以确保合并后有足够数量
        k_neighbors=ub_k_neighbors
    )

    # 获取 Content-Based CF 的推荐分数
    cb_recs_with_scores = recommend_content_based(
        user_id=user_id,
        user_item_ratings_train=user_item_ratings_train,
        artist_to_tags=artist_to_tags,
        unique_tags=unique_tags,
        artist_id_to_idx=artist_id_to_idx_train,
        idx_to_artist_id=idx_to_artist_id_train,
        num_recommendations=num_recommendations * 2,  # 临时获取更多
        min_tags_per_artist=min_tags_per_artist  # 传递新的参数
    )

    final_scores = defaultdict(float)
    all_recommended_artists = set()

    # 标准化分数：将所有分数映射到 0-1 之间，以便加权融合
    # 对 User-Based CF 分数进行标准化
    ub_scores_only = [score for _, score in ub_recs_with_scores]
    min_ub_score = min(ub_scores_only) if ub_scores_only else 0
    max_ub_score = max(ub_scores_only) if ub_scores_only else 0

    ub_norm_factor = max_ub_score - min_ub_score if max_ub_score - min_ub_score > 0 else 1.0

    # 对 Content-Based 分数进行标准化
    cb_scores_only = [score for _, score in cb_recs_with_scores]
    min_cb_score = min(cb_scores_only) if cb_scores_only else 0
    max_cb_score = max(cb_scores_only) if cb_scores_only else 0

    cb_norm_factor = max_cb_score - min_cb_score if max_cb_score - min_cb_score > 0 else 1.0

    # 合并 User-Based CF 结果
    for artist_id, score in ub_recs_with_scores:
        normalized_score = (score - min_ub_score) / ub_norm_factor if ub_norm_factor != 0 else 0.0
        final_scores[artist_id] += (1 - cb_weight) * normalized_score
        all_recommended_artists.add(artist_id)

    # 合并 Content-Based CF 结果
    for artist_id, score in cb_recs_with_scores:
        normalized_score = (score - min_cb_score) / cb_norm_factor if cb_norm_factor != 0 else 0.0
        final_scores[artist_id] += cb_weight * normalized_score
        all_recommended_artists.add(artist_id)

    # 获取用户已经听过的艺术家，以便最终推荐时排除
    user_listened_artists = set(user_item_ratings_train.get(user_id, {}).keys())

    # 排序并返回 Top-N 推荐 (排除已听过的)
    sorted_final_recs = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    final_recommendations = []
    for artist_id, score in sorted_final_recs:
        if artist_id not in user_listened_artists:
            final_recommendations.append(artist_id)
        if len(final_recommendations) >= num_recommendations:
            break

    return final_recommendations
