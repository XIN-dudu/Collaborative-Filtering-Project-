import numpy as np
import pandas as pd # 确保导入 pandas，因为 evaluate_model 中会用到 test_df 和 artist_info_df

def calculate_precision_recall(recommended_items, relevant_items, k):
    """
    计算推荐列表的 Precision@K 和 Recall@K。

    Args:
        recommended_items (list): 推荐系统为用户生成的 K 个推荐物品的列表（艺术家ID）。
        relevant_items (set): 用户在测试集中实际交互过的物品的集合（艺术家ID）。
        k (int): 用于计算 Precision 和 Recall 的推荐列表长度。

    Returns:
        tuple: (precision_at_k, recall_at_k)
    """
    if not recommended_items:
        return 0.0, 0.0

    # 确保推荐列表长度不超过 K
    recommended_at_k = recommended_items[:k]

    # 计算相关且被推荐的物品数量
    hits = 0
    for item in recommended_at_k:
        if item in relevant_items:
            hits += 1

    precision = hits / len(recommended_at_k) if len(recommended_at_k) > 0 else 0.0
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall

def evaluate_model(recommendation_function, test_df, user_artist_matrix_train,
                   user_similarity_matrix=None, item_similarity_matrix=None,
                   num_recommendations=10): # 移除了 artist_info_df 参数，因为推荐函数不再需要它
    """
    评估推荐模型的整体性能，计算所有用户的平均 Precision@K 和 Recall@K。

    Args:
        recommendation_function (function): 推荐生成函数 (例如 recommend_user_based_cf 或 recommend_item_based_cf)。
                                            这个函数需要接收 user_id, user_artist_matrix,
                                            以及对应的相似度矩阵 (user_similarity_matrix 或 item_similarity_matrix),
                                            num_recommendations 作为参数。
        test_df (pd.DataFrame): 测试集数据，包含 'userID', 'artistID', 'weight'。
        user_artist_matrix_train (pd.DataFrame): 训练集的用户-艺术家矩阵。
        user_similarity_matrix (pd.DataFrame, optional): 用户相似度矩阵 (如果使用 User-Based CF)。
        item_similarity_matrix (pd.DataFrame, optional): 物品相似度矩阵 (如果使用 Item-Based CF)。
        num_recommendations (int): 用于评估的推荐数量 K。

    Returns:
        tuple: (average_precision, average_recall)
    """
    total_precision = 0.0
    total_recall = 0.0
    evaluated_users_count = 0

    # 找到测试集中所有唯一的用户
    test_users = test_df['userID'].unique()

    for user_id in test_users:
        # 获取用户在测试集中实际听过的艺术家
        relevant_artists_in_test = set(test_df[test_df['userID'] == user_id]['artistID'].tolist())

        # 如果测试集中该用户没有实际听过的艺术家，或者该用户在训练集中不存在，则跳过
        if not relevant_artists_in_test or user_id not in user_artist_matrix_train.index:
            continue

        # 生成推荐：现在推荐函数直接返回 artistID 列表
        recommended_artist_ids = []
        if user_similarity_matrix is not None:
            # 移除 artist_info_df 参数的传递
            recommended_artist_ids = recommendation_function(
                user_id=user_id,
                user_artist_matrix=user_artist_matrix_train,
                user_similarity_matrix=user_similarity_matrix,
                num_recommendations=num_recommendations
            )
        elif item_similarity_matrix is not None:
            # 移除 artist_info_df 参数的传递
            recommended_artist_ids = recommendation_function(
                user_id=user_id,
                user_artist_matrix=user_artist_matrix_train,
                item_similarity_matrix=item_similarity_matrix,
                num_recommendations=num_recommendations
            )
        else:
            # 这个警告信息可以保留，但理论上不应该触发
            print(f"警告: 用户 {user_id} 未能生成推荐，因为没有提供相似度矩阵。")
            continue

        # 这一段将“推荐的艺术家名称转换为ID”的逻辑现在是多余的，因为推荐函数已经返回ID了
        # 因此，这里直接使用 recommended_artist_ids 即可
        # 移除以下被注释掉的代码块:
        # recommended_artist_ids_from_names = []
        # for artist_name in recommended_artist_names:
        #     matching_artists = artist_info_df[artist_info_df['artistName'] == artist_name]['artistID']
        #     if not matching_artists.empty:
        #         recommended_artist_ids_from_names.append(matching_artists.iloc[0])
        #     else:
        #         pass

        if not recommended_artist_ids: # 确保推荐列表不为空
            # 如果没有生成任何推荐，则跳过该用户
            continue

        precision, recall = calculate_precision_recall(recommended_artist_ids, relevant_artists_in_test,
                                                       num_recommendations)

        total_precision += precision
        total_recall += recall
        evaluated_users_count += 1

    if evaluated_users_count > 0:
        average_precision = total_precision / evaluated_users_count
        average_recall = total_recall / evaluated_users_count
    else:
        average_precision = 0.0
        average_recall = 0.0

    return average_precision, average_recall


# 调试/测试用 (非正式运行，仅用于检查函数定义)
if __name__ == '__main__':
    print("recommender_eval.py 模块已加载。")
    # 你可以在这里添加一些小型的测试用例来验证你的函数
    # 例如：
    # rec = [1, 2, 3, 4, 5]
    # rel = {1, 3, 6, 7}
    # p, r = calculate_precision_recall(rec, rel, 5)
    # print(f"Precision: {p}, Recall: {r}")