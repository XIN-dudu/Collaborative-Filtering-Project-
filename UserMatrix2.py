import numpy as np
import pandas as pd
import collections

# --- 修改后的 build_user_artist_matrix 函数 ---
def build_user_artist_matrix(
    user_artists_path='resources/user_artists.dat',
    artists_path='resources/artists.dat',
    min_listen_count=10,
    min_users_per_artist=5,
    min_artists_per_user=10
):
    """
    读取 Last.fm 的 user_artists.dat 和 artists.dat 文件，
    构建用户-艺术家矩阵，并进行初步过滤。
    ... (函数文档保持不变) ...
    """

    print(f"开始构建用户-艺术家矩阵...")

    # 1. 读取 user_artists.dat - 这是核心数据
    try:
        df_ua = pd.read_csv(
            user_artists_path,
            sep='\t',
            header=None,
            names=['userID', 'artistID', 'weight']
        )
        print(f"原始 user_artists.dat 加载完成，共有 {len(df_ua)} 条记录。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {user_artists_path}。请检查路径。")
        return pd.DataFrame(), pd.DataFrame()

    # >>>>>>>>>> 在这里添加类型转换代码 <<<<<<<<<<
    # 强制将 'weight' 列转换为数值类型。
    # errors='coerce' 会将任何无法转换的值设为 NaN。
    df_ua['weight'] = pd.to_numeric(df_ua['weight'], errors='coerce')

    # 删除因为类型转换失败而产生的 NaN 行（即那些非数字的 'weight' 值）
    df_ua.dropna(subset=['weight'], inplace=True)
    # 也可以选择将 NaN 填充为 0 或其他值：
    # df_ua['weight'].fillna(0, inplace=True)
    # 但对于 'weight' (收听次数)，直接删除无效记录更合理。

    # 确保 'weight' 列是整数类型（如果需要）
    df_ua['weight'] = df_ua['weight'].astype(int)
    # >>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<

    # 2. 读取 artists.dat - 用于获取艺术家名称，方便推荐结果展示
    try:
        df_artists = pd.read_csv(
            artists_path,
            sep='\t',
            header=None,
            names=['artistID', 'artistName', 'artistURL', 'pictureURL']
        )
        print(f"原始 artists.dat 加载完成，共有 {len(df_artists)} 条记录。")
    except FileNotFoundError:
        print(f"警告: 找不到文件 {artists_path}。将不提供艺术家名称映射。")
        df_artists = pd.DataFrame(columns=['artistID', 'artistName'])

    # 3. 数据过滤 (根据你的需求和 readme.txt 描述)
    # 过滤掉收听次数过低的记录
    df_ua = df_ua[df_ua['weight'] >= min_listen_count]
    print(f"过滤掉收听次数低于 {min_listen_count} 的记录后，剩余 {len(df_ua)} 条记录。")

    # 过滤掉不活跃用户 (听歌艺术家数量过少的用户)
    user_counts = df_ua.groupby('userID')['artistID'].count()
    active_users = user_counts[user_counts >= min_artists_per_user].index
    df_ua = df_ua[df_ua['userID'].isin(active_users)]
    print(f"过滤掉听歌艺术家数量低于 {min_artists_per_user} 的用户后，剩余 {df_ua['userID'].nunique()} 位用户。")

    # 过滤掉不流行艺术家 (被听用户数量过少的艺术家)
    artist_listener_counts = df_ua.groupby('artistID')['userID'].count()
    popular_artists = artist_listener_counts[artist_listener_counts >= min_users_per_artist].index
    df_ua = df_ua[df_ua['artistID'].isin(popular_artists)]
    print(f"过滤掉被听用户数量低于 {min_users_per_artist} 的艺术家后，剩余 {df_ua['artistID'].nunique()} 位艺术家。")

    # 4. 构建用户-艺术家矩阵 (Pivot Table)
    user_artist_matrix = df_ua.pivot_table(
        index='userID',
        columns='artistID',
        values='weight',
        fill_value=0
    )
    print(f"用户-艺术家矩阵构建完成，维度：{user_artist_matrix.shape} (用户数 x 艺术家数)")
    print("矩阵示例 (前5行5列):")
    print(user_artist_matrix.iloc[:5, :5])

    # 5. 准备艺术家名称映射表
    artist_names_map = df_artists.set_index('artistID')['artistName'].to_dict()
    df_artist_info = df_artists[['artistID', 'artistName']]

    print("用户-艺术家矩阵和艺术家信息已准备好。")
    return user_artist_matrix, df_artist_info