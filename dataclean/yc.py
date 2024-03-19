import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 读取数据集
data = pd.read_csv('../2022-property-sales-data备份.csv')

data.drop(columns='Extwall', inplace=True)
columns_to_fill = [col for col in data.columns if col not in ['PropType', 'RICH', 'Style']]
data = pd.get_dummies(data, columns=columns_to_fill)

for col in columns_to_fill:
    # 分离已知和未知数据
    known_data = data[data[col].notnull()]
    unknown_data = data[data[col].isnull()]

    # 分离特征和目标列
    X_known = known_data.drop(columns=columns_to_fill)
    y_known = known_data[col]
    X_unknown = unknown_data.drop(columns=columns_to_fill)

    # 使用随机森林回归填充数值类型的列，分类填充非数值类型的列
    if data[col].dtype != 'object':  # 数值类型列
        model = RandomForestRegressor()
    else:  # 非数值类型列
        model = RandomForestClassifier()

    model.fit(X_known, y_known)
    predicted_values = model.predict(X_unknown)

    # 填充预测值
    data.loc[data[col].isnull(), col] = predicted_values

# 显示填充后的数据集
print(data.head())
