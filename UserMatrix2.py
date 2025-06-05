# UserMatrix2.py

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np  # 确保引入 numpy


def build_user_artist_matrix(user_artists_path, artists_path, min_listen_count,
                             min_users_per_artist, min_artists_per_user, input_df=None):
    """
    从原始数据或提供的DataFrame构建用户-艺术家CSR矩阵。
    同时返回ID到索引的映射以及用户-艺术家播放次数的字典。

    参数:
        user_artists_path (str): user_artists.dat 文件路径。如果 input_df 不为 None，则忽略。
        artists_path (str): artists.dat 文件路径（用于获取艺术家名称，如果需要）。
        min_listen_count (int): 最低播放次数过滤。
        min_users_per_artist (int): 艺术家被听过的最低用户数过滤。
        min_artists_per_user (int): 用户听过的最低艺术家数过滤。
        input_df (pd.DataFrame, optional): 如果提供了，则直接使用此DataFrame作为输入数据。

    返回:
        tuple: (user_artist_matrix (csr_matrix), user_id_to_idx (dict),
                idx_to_user_id (dict), artist_id_to_idx (dict),
                idx_to_artist_id (dict), user_item_ratings (dict))
        如果构建失败，可能返回 None 或空结构。
    """
    if input_df is not None:
        df_user_artists = input_df
    else:
        try:
            df_user_artists = pd.read_csv(user_artists_path, sep='\t')
        except FileNotFoundError:
            print(f"错误: 未找到文件 {user_artists_path}")
            return None, None, None, None, None, None

    # 1. 过滤低听歌次数记录
    filtered_df = df_user_artists[df_user_artists['weight'] >= min_listen_count].copy()

    # 2. 过滤艺术家：听过该艺术家的用户数量 >= min_users_per_artist
    artist_counts = filtered_df['artistID'].value_counts()
    valid_artists = artist_counts[artist_counts >= min_users_per_artist].index
    filtered_df = filtered_df[filtered_df['artistID'].isin(valid_artists)]

    # 3. 过滤用户：听过艺术家数量 >= min_artists_per_user
    user_artist_counts = filtered_df.groupby('userID')['artistID'].nunique()
    valid_users = user_artist_counts[user_artist_counts >= min_artists_per_user].index
    filtered_df = filtered_df[filtered_df['userID'].isin(valid_users)]

    if filtered_df.empty:
        print("警告: 经过过滤后，DataFrame 为空。请检查过滤条件。")
        # 返回空矩阵和空映射，确保返回类型一致
        return csr_matrix((0, 0)), {}, {}, {}, {}, {}

    # 构建 ID 到索引的映射
    unique_users = filtered_df['userID'].unique()
    unique_artists = filtered_df['artistID'].unique()

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
    artist_id_to_idx = {artist_id: idx for idx, artist_id in enumerate(unique_artists)}
    idx_to_artist_id = {idx: artist_id for artist_id, idx in artist_id_to_idx.items()}

    # 构建用户-艺术家播放次数的字典 (用于评估和推荐时过滤已听歌曲)
    user_item_ratings = {}
    for _, row in filtered_df.iterrows():
        user_id = row['userID']
        artist_id = row['artistID']
        play_count = row['weight']
        user_item_ratings.setdefault(user_id, {})[artist_id] = play_count

    # 构建稀疏矩阵 (CSR 格式)
    rows = []
    cols = []
    data = []
    for _, row in filtered_df.iterrows():
        user_idx = user_id_to_idx[row['userID']]
        artist_idx = artist_id_to_idx[row['artistID']]
        rows.append(user_idx)
        cols.append(artist_idx)
        data.append(row['weight'])

    user_artist_matrix = csr_matrix((data, (rows, cols)),
                                    shape=(len(unique_users), len(unique_artists)))

    # 返回 CSR 矩阵和所有映射以及 user_item_ratings 字典
    return user_artist_matrix, user_id_to_idx, idx_to_user_id, \
        artist_id_to_idx, idx_to_artist_id, user_item_ratings


def preprocess_tag_data(user_tagged_artists_df, artist_id_to_idx, tag_id_to_value):
    """
    预处理标签数据，构建艺术家-标签映射。

    Args:
        user_tagged_artists_df (pd.DataFrame): 从 user_taggedartists.dat 加载的 DataFrame。
        artist_id_to_idx (dict): 艺术家ID到矩阵索引的映射 (来自训练集)。
        tag_id_to_value (dict): 标签ID到标签值的映射。

    Returns:
        dict: 艺术家ID到其关联标签列表的映射，例如 {artist_id: [tag_value1, tag_value2, ...]}。
        dict: 艺术家ID到其关联标签ID列表的映射，例如 {artist_id: [tag_id1, tag_id2, ...]}。
        set: 所有唯一的标签值集合。
    """
    artist_to_tags = {}
    artist_to_tag_ids = {}
    unique_tags = set()

    # 过滤掉不在训练集艺术家列表中的标签记录
    filtered_tags_df = user_tagged_artists_df[
        user_tagged_artists_df['artistID'].isin(artist_id_to_idx.keys())
    ]

    for _, row in filtered_tags_df.iterrows():
        artist_id = int(row['artistID'])
        tag_id = int(row['tagID'])

        # 确保标签ID存在于 tags.dat 中
        if tag_id in tag_id_to_value:
            tag_value = tag_id_to_value[tag_id]

            artist_to_tags.setdefault(artist_id, []).append(tag_value)
            artist_to_tag_ids.setdefault(artist_id, []).append(tag_id)
            unique_tags.add(tag_value)

    # 去重：一个艺术家可能被不同用户打上相同的标签
    for artist_id in artist_to_tags:
        artist_to_tags[artist_id] = list(set(artist_to_tags[artist_id]))
        artist_to_tag_ids[artist_id] = list(set(artist_to_tag_ids[artist_id]))

    print(
        f"标签数据预处理完成。从 {len(filtered_tags_df)} 条记录中提取了 {len(artist_to_tags)} 位艺术家的标签，共 {len(unique_tags)} 个唯一标签。")
    return artist_to_tags, artist_to_tag_ids, unique_tags

