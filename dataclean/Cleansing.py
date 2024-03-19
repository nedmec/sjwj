import pandas as pd

# 读取CSV文件
file_path = '../cleaned_file.csv'  # 将'your_file.csv'替换为实际的文件路径
df = pd.read_csv(file_path, header=None)

# 检查是否存在空缺值
has_missing_values = df.isnull().values.any()

if has_missing_values:
    # 输出包含空缺值的样本
    rows_with_missing_values = df[df.isnull().any(axis=1)]
    print("存在空缺值的样本:")
    print(rows_with_missing_values)
    print(f"共有 {len(rows_with_missing_values)} 个样本包含空缺值。")
else:
    print("文件中没有空缺值。")