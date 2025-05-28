# recommender_eval.py

import numpy as np


def evaluate_model(recommendation_function, user_item_ratings_test, users_to_evaluate_ids, top_k=10):
    """
    评估推荐模型的 Precision@K 和 Recall@K。
    Args:
        recommendation_function (function): 一个接受 user_id 并返回推荐艺术家ID列表的函数。
                                            这个函数应该在 main 程序中通过 lambda 表达式封装好所有必需的参数。
        user_item_ratings_test (dict): 测试集中用户实际听过的艺术家及其播放次数，格式为
                                       {user_id: {artist_id: play_count, ...}}.
        users_to_evaluate_ids (list): 实际要进行评估的用户ID列表。
        top_k (int): 评估 Top-K 推荐。
    Returns:
        tuple: (average_precision, average_recall)
    """
    total_precision = 0.0
    total_recall = 0.0
    num_evaluated_users = 0

    print(f"开始评估 {len(users_to_evaluate_ids)} 个用户...")

    for user_id in users_to_evaluate_ids:
        actual_items = set(user_item_ratings_test.get(user_id, {}).keys())

        # 如果用户在测试集中没有实际的听歌记录，则跳过
        if not actual_items:
            # print(f"警告: 用户 {user_id} 在测试集中没有实际听歌记录，跳过评估。") # 调试时可以打开
            continue

        # 调用推荐函数获取推荐列表
        # recommendation_function 现在是一个包装过的 lambda，它内部已经有了所有必要的参数
        recommended_items = recommendation_function(user_id)

        if not recommended_items:
            # print(f"警告: 未能为用户 {user_id} 生成任何推荐。") # 调试时可以打开
            # 如果没有生成推荐，Precision 和 Recall 都为 0
            # 仍然计入总数以反映模型未能提供推荐的情况
            total_precision += 0.0
            total_recall += 0.0
            num_evaluated_users += 1
            continue

        # 计算 Precision@K
        # 推荐列表中实际相关的项目数量
        hits = len(actual_items.intersection(set(recommended_items)))
        precision = hits / len(recommended_items)
        total_precision += precision

        # 计算 Recall@K
        recall = hits / len(actual_items)
        total_recall += recall

        num_evaluated_users += 1

    if num_evaluated_users == 0:
        print("没有可用于评估的用户，返回 0.0。")
        return 0.0, 0.0

    average_precision = total_precision / num_evaluated_users
    average_recall = total_recall / num_evaluated_users

    return average_precision, average_recall