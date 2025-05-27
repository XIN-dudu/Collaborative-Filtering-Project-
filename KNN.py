import pandas as pd
import numpy as np


# from sklearn.preprocessing import MinMaxScaler # 如果需要归一化，可以保留

# --- 距离和相似度计算函数（保留，尽管我们主要用 sklearn 的 cosine_similarity）---

def euclidean_distance(vec1, vec2):
    """
    计算两个向量之间的欧几里得距离。
    Args:
        vec1 (np.array or list): 第一个向量。
        vec2 (np.array or list): 第二个向量。
    Returns:
        float: 欧几里得距离。
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def cosine_similarity_custom(vec1, vec2):
    """
    计算两个向量之间的余弦相似度（自定义实现）。
    Args:
        vec1 (np.array or list): 第一个向量。
        vec2 (np.array or list): 第二个向量。
    Returns:
        float: 余弦相似度。
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def pearson_correlation(vec1, vec2):
    """
    计算两个向量之间的皮尔逊相关系数。
    Args:
        vec1 (np.array or list): 第一个向量。
        vec2 (np.array or list): 第二个向量。
    Returns:
        float: 皮尔逊相关系数。
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # 确保向量长度一致且不为0
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return 0.0

    mean_vec1 = np.mean(vec1)
    mean_vec2 = np.mean(vec2)

    numerator = np.sum((vec1 - mean_vec1) * (vec2 - mean_vec2))
    denominator = np.sqrt(np.sum((vec1 - mean_vec1) ** 2) * np.sum((vec2 - mean_vec2) ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def manhattan_distance(vec1, vec2):
    """
    计算两个向量之间的曼哈顿距离。
    Args:
        vec1 (np.array or list): 第一个向量。
        vec2 (np.array or list): 第二个向量。
    Returns:
        float: 曼哈顿距离。
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    return np.sum(np.abs(vec1 - vec2))


# scikit-learn 的 jaccard_similarity_score 在新版本中已弃用，
# 且更适用于二值特征。如果需要，可以使用 scipy.spatial.distance.jaccard
# 或者自己实现，例如:
# def jaccard_coefficient(set1, set2):
#    intersection = len(set1.intersection(set2))
#    union = len(set1.union(set2))
#    return intersection / union if union != 0 else 0.0


# --- K 近邻查找核心函数 ---

def find_k_nearest_neighbors(query_id, similarity_matrix, k_neighbors, exclude_self=True):
    """
    从相似度矩阵中找到指定查询ID的 K 个最近邻居。

    Args:
        query_id: 要查找邻居的ID (用户ID或物品ID)。
        similarity_matrix (pd.DataFrame): 相似度矩阵 (例如 user_similarity_matrix 或 item_similarity_matrix)。
        k_neighbors (int): 要返回的最近邻居数量。
        exclude_self (bool): 是否排除查询ID本身。

    Returns:
        pd.Series: 包含 K 个最近邻居及其相似度分数的 Series。
                   Series 的索引是邻居的ID，值是相似度。
                   按相似度降序排列。
    """
    if query_id not in similarity_matrix.index:
        return pd.Series([], dtype='float64')  # 返回一个空的 Series

    # 获取与 query_id 的所有相似度
    similarities = similarity_matrix[query_id]

    if exclude_self and query_id in similarities.index:
        similarities = similarities.drop(query_id)  # 排除自己

    # 筛选出相似度大于0的邻居，并选择最高的 k_neighbors 个
    # .nlargest() 方法可以高效地获取最大的 N 个值
    k_nearest = similarities[similarities > 0].nlargest(k_neighbors)

    return k_nearest


# 调试/测试用
if __name__ == '__main__':
    print("KNN.py 模块已加载。")
