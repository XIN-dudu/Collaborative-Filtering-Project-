import pandas as pd
import numpy as np
import KNN  # 导入你的 KNN 模块


def recommend_user_based_cf(user_id, user_artist_matrix, user_similarity_matrix, num_recommendations=10,
                            k_neighbors=50):
    """
    为指定用户生成基于用户的协同过滤推荐 (使用 KNN 模块优化)。

    Args:
        user_id (int): 目标用户的 ID。
        user_artist_matrix (pd.DataFrame): 用户-艺术家收听矩阵 (训练集)。
        user_similarity_matrix (pd.DataFrame): 用户相似度矩阵。
        num_recommendations (int): 推荐艺术家的数量。
        k_neighbors (int): 考虑用于预测的最近邻居数量。

    Returns:
        list: 推荐的艺术家ID列表。
    """
    if user_id not in user_artist_matrix.index:
        return []

    user_listened_artists = user_artist_matrix.loc[user_id][user_artist_matrix.loc[user_id] > 0].index.tolist()

    # 使用 KNN 模块的 find_k_nearest_neighbors 获取 K 个最近邻居
    similar_users = KNN.find_k_nearest_neighbors(
        query_id=user_id,
        similarity_matrix=user_similarity_matrix,
        k_neighbors=k_neighbors,
        exclude_self=True
    )

    # 确保相似用户在训练矩阵的索引中
    similar_users = similar_users[similar_users.index.isin(user_artist_matrix.index)]

    predicted_ratings = {}
    # 遍历所有艺术家，寻找用户未听过的
    for artist_id in user_artist_matrix.columns:
        if artist_id not in user_listened_artists:
            weighted_sum = 0
            similarity_sum = 0

            # 遍历 K 个最近的相似用户
            for sim_user, similarity in similar_users.items():
                # 仅当相似用户实际听过该艺术家时才计入
                if sim_user in user_artist_matrix.index and \
                        artist_id in user_artist_matrix.columns and \
                        user_artist_matrix.loc[sim_user, artist_id] > 0:
                    weighted_sum += similarity * user_artist_matrix.loc[sim_user, artist_id]
                    similarity_sum += similarity

            if similarity_sum > 0:
                predicted_ratings[artist_id] = weighted_sum / similarity_sum
            else:
                predicted_ratings[artist_id] = 0  # 无法预测，给个0分

    recommended_artist_ids = sorted(predicted_ratings.items(), key=lambda item: item[1], reverse=True)[
                             :num_recommendations]

    return [artist_id for artist_id, _ in recommended_artist_ids]


def recommend_item_based_cf(user_id, user_artist_matrix, item_similarity_matrix, num_recommendations=10,
                            k_neighbors=50):
    """
    为指定用户生成基于物品的协同过滤推荐 (使用 KNN 模块优化)。

    Args:
        user_id (int): 目标用户的 ID。
        user_artist_matrix (pd.DataFrame): 用户-艺术家收听矩阵 (训练集)。
        item_similarity_matrix (pd.DataFrame): 物品相似度矩阵。
        num_recommendations (int): 推荐艺术家的数量。
        k_neighbors (int): 考虑用于预测的最近邻居数量。

    Returns:
        list: 推荐的艺术家ID列表。
    """
    if user_id not in user_artist_matrix.index:
        return []

    # 获取目标用户已听过的艺术家ID及其收听次数
    user_listened_artists = user_artist_matrix.loc[user_id][user_artist_matrix.loc[user_id] > 0]

    predicted_ratings = {}

    all_artists = user_artist_matrix.columns
    for candidate_artist_id in all_artists:
        # 确保候选艺术家不在用户已听列表中，且在物品相似度矩阵的索引中
        if candidate_artist_id not in user_listened_artists.index and \
                candidate_artist_id in item_similarity_matrix.index:

            weighted_sum = 0
            similarity_sum = 0

            # 筛选出用户实际听过的艺术家，且这些艺术家在物品相似度矩阵的列中
            listened_and_in_sim_matrix_artists = [
                art_id for art_id in user_listened_artists.index
                if art_id in item_similarity_matrix.columns
            ]

            if not listened_and_in_sim_matrix_artists:
                predicted_ratings[candidate_artist_id] = 0
                continue

            # 使用 KNN 模块的 find_k_nearest_neighbors 获取与候选艺术家最相似的 K 个已听艺术家
            # 注意：这里的相似度矩阵是 item_similarity_matrix，查询对象是 candidate_artist_id
            # 并且我们只在用户已听过的艺术家子集中寻找相似邻居

            # 临时构建一个只包含用户已听艺术家的子相似度 Series
            # 这样 find_k_nearest_neighbors 可以在这个子集中查找
            temp_sim_series = item_similarity_matrix.loc[candidate_artist_id, listened_and_in_sim_matrix_artists]

            # 传入一个临时的 DataFrame 或者直接操作 Series
            # 更直接的方式是直接从 item_similarity_matrix 中筛选并排序

            similar_listened_items_top_k = temp_sim_series[temp_sim_series > 0].nlargest(k_neighbors)

            for listened_artist_id, similarity in similar_listened_items_top_k.items():
                # 获取用户对该已听艺术家的收听次数
                # 确保 listened_artist_id 存在于 user_listened_artists
                if listened_artist_id in user_listened_artists.index:
                    listen_count = user_listened_artists.loc[listened_artist_id]
                    weighted_sum += similarity * listen_count
                    similarity_sum += similarity

            if similarity_sum > 0:
                predicted_ratings[candidate_artist_id] = weighted_sum / similarity_sum
            else:
                predicted_ratings[candidate_artist_id] = 0  # 无法预测，给个0分

    recommended_artist_ids = sorted(predicted_ratings.items(), key=lambda item: item[1], reverse=True)[
                             :num_recommendations]

    return [artist_id for artist_id, _ in recommended_artist_ids]


# 调试/测试用 (非正式运行，仅用于检查函数定义)
if __name__ == '__main__':
    print("recommender_algorithms.py 模块已加载。")