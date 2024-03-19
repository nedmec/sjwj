import pandas as pd

# 假设您的数据存储在名为 'your_data.csv' 的 CSV 文件中
# 如果实际文件名不同，请将 'your_data.csv' 替换为实际文件名
file_path = '../2022-property-sales-data备份.csv'

# 将 CSV 文件读入 DataFrame
df = pd.read_csv(file_path)

# 删除包含空值的行
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates(subset=df_cleaned.columns[0])
df_cleaned = df_cleaned.drop(df.columns[4], axis=1)
# 打印清理后的 DataFrame
print(df_cleaned)

# 将清理后的 DataFrame 保存回 CSV 文件
cleaned_file_path = '../cleaned_data.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"已将清理后的数据保存到 '{cleaned_file_path}' 文件中。")