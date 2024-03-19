import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
file_path = './cleaned_data.csv'
font = FontProperties(fname='SimHei.ttf', size=12)
# 将 CSV 文件读入 DataFrame
df = pd.read_csv(file_path)
# 假设你的数据已经读取到 DataFrame 中，命名为 df
# 你需要先将非数值型数据进行编码
non_numeric_columns = df.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for column in non_numeric_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# 确保所有列都是数值型的
numeric_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()

# 对数据进行标准化，如果需要的话
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 使用 PCA
pca = PCA()
pca.fit(df[numeric_columns])

explained_var_ratio = pca.explained_variance_ratio_
cumulative_explained_var_ratio = np.cumsum(explained_var_ratio)

# 绘制累计解释方差图
import matplotlib.pyplot as plt
plt.plot(cumulative_explained_var_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# 根据累计解释方差，确定保留的主成分数量
desired_variance = 0.95  # 设定希望保留的方差解释比例
n_components = np.argmax(cumulative_explained_var_ratio >= desired_variance) + 1

print(f"保留 {desired_variance * 100:.2f}% 方差所需的主成分数量为：{n_components}")
