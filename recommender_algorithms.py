# recommender_algorithms.py

import numpy as np
from scipy.sparse import csr_matrix  # 确保导入 csr_matrix
from sklearn.metrics.pairwise import cosine_similarity  # , pearson_similarity # 引入 pearson_similarity 如果要尝试


# --- 现有函数 (如果你的文件中有，请保持一致，否则删除) ---

def calculate_user_similarity_cosine(user_artist_matrix):
    """
    计算用户之间的余弦相似度。
    Args:
        user_artist_matrix (csr_matrix): 用户-艺术家交互矩阵。
    Returns:
        np.array: 稠密的用户相似度矩阵。
    """
    # 确保输入是 csr_matrix
    if not isinstance(user_artist_matrix, csr_matrix):
        # 如果不是，尝试转换为 csr_matrix，这可能是一个错误情况
        # 你的主程序应该确保传递 csr_matrix
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


# --- 新增/修改函数 ---

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
        list: 推荐的艺术家 ID 列表。
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

    # 按预测评分排序并返回 Top-N 推荐
    recommended_artists = sorted(artist_scores.items(), key=lambda item: item[1], reverse=True)
    return [artist_id for artist_id, score in recommended_artists[:num_recommendations]]


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
        list: 推荐的艺术家 ID 列表。
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

    # 按预测评分排序并返回 Top-N 推荐
    recommended_artists = sorted(artist_scores.items(), key=lambda item: item[1], reverse=True)
    return [artist_id for artist_id, score in recommended_artists[:num_recommendations]]