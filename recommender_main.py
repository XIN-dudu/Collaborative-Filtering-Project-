import pandas as pd
import numpy as np
import UserMatrix2 # 导入我们修改过的 UserMatrix2 模块

# 定义数据文件路径（确保与你的实际路径匹配）
# 如果 UserMatrix2.py 中的函数参数已经有默认值指向 'resources/'，
# 那么这里可以省略，但在 main 脚本中显式指定路径是个好习惯，方便管理。
USER_ARTISTS_DAT_PATH = 'resources/user_artists.dat'
ARTISTS_DAT_PATH = 'resources/artists.dat'

print("--- 音乐推荐系统启动 ---")
print("阶段 1: 数据加载与用户-艺术家矩阵构建")

# 调用 UserMatrix2 中修改过的函数来构建矩阵
user_artist_matrix, artist_info_df = UserMatrix2.build_user_artist_matrix(
    user_artists_path=USER_ARTISTS_DAT_PATH,
    artists_path=ARTISTS_DAT_PATH,
    min_listen_count=10,        # 艺术家被听的最低次数
    min_users_per_artist=5,     # 艺术家至少被听的用户数
    min_artists_per_user=10     # 用户至少听过的艺术家数
)

if not user_artist_matrix.empty:
    print("\n--- 阶段 1 完成：用户-艺术家矩阵成功构建 ---")
    print("矩阵维度:", user_artist_matrix.shape)
    # 打印矩阵的前几行几列，进行目视检查
    print("矩阵示例（前5行5列）：")
    print(user_artist_matrix.iloc[:5, :5])

    # 保存构建好的矩阵到CSV（可选，方便后续调试和检查）
    # user_artist_matrix.to_csv('output_user_artist_matrix.csv')
    # artist_info_df.to_csv('output_artist_info.csv')
    # print("\n用户-艺术家矩阵和艺术家信息已保存到CSV文件。")

    # --- 接下来将是协同过滤算法的核心实现部分 ---
    print("\n--- 准备进入阶段 2: 协同过滤算法实现 ---")
    # 这里将是你的核心推荐逻辑、相似度计算、推荐生成和评估代码
    # 在你确认阶段1完全成功后，再继续填充这里。

else:
    print("\n--- 阶段 1 失败：用户-艺术家矩阵构建失败 ---")
    print("请检查文件路径和数据格式，或查看 UserMatrix2.py 的错误信息。")

print("\n--- 音乐推荐系统关闭 ---")