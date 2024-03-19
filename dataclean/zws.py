import pandas as pd

# 假设您的数据存储在名为 'your_data.csv' 的 CSV 文件中
# 如果实际文件名不同，请将 'your_data.csv' 替换为实际文件名
file_path = '../2022-property-sales-data备份.csv'

# 将 CSV 文件读入 DataFrame
df = pd.read_csv(file_path, header=None)  # 假设没有列名
df = df.drop(df.columns[4], axis=1)
# 保存原始字符串列
string_columns = [1, 3, 13]
original_strings = df.iloc[:, string_columns].copy()



# 对字符串型列进行哑变量编码
df_encoded = pd.get_dummies(df, columns=string_columns, prefix=string_columns)

# 使用中位数填充剩余的缺失值

numeric_columns = df_encoded.select_dtypes(include='number').columns
df_encoded[numeric_columns] = df_encoded[numeric_columns].fillna(df_encoded[numeric_columns].median())

# 将编码后的数据逆映射回原始字符串
for column in string_columns:
    col_mapping = {f"{column}_{val}": val for val in original_strings[column].unique()}
    df_encoded[column] = df_encoded.apply(lambda row: col_mapping.get(row[f"{column}_1"], row[f"{column}_1"]), axis=1)
# 打印清理后的 DataFrame
print(df_encoded)

# 将清理后的 DataFrame 保存回 CSV 文件
cleaned_file_path = '../zws_cleaned_data.csv'
df_encoded.to_csv(cleaned_file_path, index=False, header=False)  # 不保存列名

print(f"已将清理后的数据保存到 '{cleaned_file_path}' 文件中。")
