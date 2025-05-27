import numpy as np
import pandas as pd
import collections

def build_user_artist_matrix(
    user_artists_path=None,
    artists_path='resources/artists.dat',
    min_listen_count=10,
    min_users_per_artist=5,
    min_artists_per_user=10,
    input_df=None
):
    """
    构建用户-艺术家矩阵。
    """
    print("开始构建用户-艺术家矩阵...")

    # 1. 加载 user_artists.dat 或使用传入的 DataFrame
    if input_df is not None:
        df_ua = input_df.copy()
        print(f"传入的 user_artists DataFrame 已加载，共有 {len(df_ua)} 条记录。")
    else:
        if user_artists_path is None:
            print("错误: 既未提供 user_artists_path 也未提供 input_df。无法构建矩阵。")
            return pd.DataFrame(), pd.DataFrame()
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
        except Exception as e:
            print(f"加载 user_artists.dat 时发生错误: {e}")
            return pd.DataFrame(), pd.DataFrame()

    df_ua['weight'] = pd.to_numeric(df_ua['weight'], errors='coerce')
    df_ua.dropna(subset=['weight'], inplace=True)
    df_ua['weight'] = df_ua['weight'].astype(int)

    # 2. 读取 artists.dat - **这里是修改的重点**
    try:
        # 根据你提供的信息：artists.dat 的列是 id, name, url, pictureURL
        # 因此我们先按实际列名加载，然后重命名
        raw_artists_df = pd.read_csv(
            artists_path,
            sep='\t',
            header=0,  # 或者直接删除这一行，因为 header 默认就是 0
            names=['id', 'name', 'url', 'pictureURL']  # 保持这个，它会覆盖文件中的标题行
        )
        print(f"原始 artists.dat 加载完成，共有 {len(raw_artists_df)} 条记录。")
        print("加载后 raw_artists_df 的前5行:")
        print(raw_artists_df.head()) # 打印原始加载的 artists_info_df

        # 重命名列以匹配项目中使用的名称
        artist_info_df = raw_artists_df.rename(columns={'id': 'artistID', 'name': 'artistName'})
        # 只保留需要的列
        artist_info_df = artist_info_df[['artistID', 'artistName']].copy()

        print("重命名并选择列后 artist_info_df 的前5行:")
        print(artist_info_df.head())

    except FileNotFoundError:
        print(f"警告: 找不到文件 {artists_path}。将不提供艺术家名称映射。")
        artist_info_df = pd.DataFrame(columns=['artistID', 'artistName'])
    except Exception as e:
        print(f"加载 artists.dat 时发生错误: {e}")
        artist_info_df = pd.DataFrame(columns=['artistID', 'artistName'])


    # 3. 数据过滤
    df_ua = df_ua[df_ua['weight'] >= min_listen_count]
    print(f"过滤掉收听次数低于 {min_listen_count} 的记录后，剩余 {len(df_ua)} 条记录。")

    user_counts = df_ua.groupby('userID')['artistID'].count()
    active_users = user_counts[user_counts >= min_artists_per_user].index
    df_ua = df_ua[df_ua['userID'].isin(active_users)]
    print(f"过滤掉听歌艺术家数量低于 {min_artists_per_user} 的用户后，剩余 {df_ua['userID'].nunique()} 位用户。")

    artist_listener_counts = df_ua.groupby('artistID')['userID'].count()
    popular_artists = artist_listener_counts[artist_listener_counts >= min_users_per_artist].index
    df_ua = df_ua[df_ua['artistID'].isin(popular_artists)]
    print(f"过滤掉被听用户数量低于 {min_users_per_artist} 的艺术家后，剩余 {df_ua['artistID'].nunique()} 位艺术家。")

    # 确保 artist_info_df 只包含有效艺术家
    # 在这里添加打印，查看过滤前的 df_ua['artistID'].unique()
    print(f"df_ua['artistID'].unique() 的数量: {len(df_ua['artistID'].unique())}")
    print(f"artist_info_df 过滤前数量: {len(artist_info_df)}") # 这里的 artist_info_df 已经是重命名后的

    # 确保只保留在 df_ua 中存在的艺术家信息
    # 并且 artist_info_df 已经有了正确的 'artistID' 列
    artist_info_df = artist_info_df[artist_info_df['artistID'].isin(df_ua['artistID'].unique())].copy()

    print(f"artist_info_df 过滤后数量: {len(artist_info_df)}")
    if artist_info_df.empty:
        print("警告: 经过过滤后，artist_info_df 变为空！这可能导致无法显示艺术家名称。")


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

    print("用户-艺术家矩阵和艺术家信息已准备好。")
    return user_artist_matrix, artist_info_df
