import pandas as pd
import os

# 路径根据之前的信息设定
file_path = os.path.join('Tenrec', 'ctr_task', 'ctr_data_1M.csv')

# 如果文件不存在，回退到当前目录找找看（兼容 Notebook 运行时的路径）
if not os.path.exists(file_path):
    file_path = 'ctr_data_1M.csv'

try:
    if os.path.exists(file_path):
        # 读取前5行，了解数据结构
        data = pd.read_csv(file_path, nrows=5)
        print("=== 数据集列名 ===")
        print(data.columns.tolist())
        print("\n=== 数据集前5行样例 ===")
        print(data)
        print("\n=== 各列数据类型 ===")
        print(data.dtypes)
    else:
        print(f"Error: 找不到文件 {file_path}")
except Exception as e:
    print(f"读取数据出错: {e}")
