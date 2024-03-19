from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import os
from models import db
from sqlalchemy import text
import base64
import io

kmeans_model = KMeans(n_clusters=3, n_init=10)
from flask import current_app


def dbscan(table_name):
    index = int(table_name[-1])
    query = text(f"SELECT * FROM 'properties' WHERE [index] = {index} ")
    table_data = db.session.execute(query).fetchall()
    column_names = [
        'PropertyID', 'PropType', 'nbhd', 'Style', 'Stories',
        'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms',
        'Fbath', 'Hbath', 'Sale_price', 'RICH', 'index'
    ]
    data = pd.DataFrame(table_data, columns=column_names)
    data = data.drop(columns=[data.columns[-1]])
    non_numeric_cols = ['PropType', 'Style', 'RICH']
    data_encoded = pd.get_dummies(data, columns=non_numeric_cols)
    scaler = StandardScaler()
    numeric_cols = ['nbhd', 'Stories', 'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms', 'Fbath', 'Hbath',
                    'Sale_price']
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols].apply(pd.to_numeric, errors='coerce'))
    data_encoded.dropna(inplace=True)
    data_processed = data_encoded.drop(['PropertyID'], axis=1)
    dbscan_model = DBSCAN(eps=6, min_samples=10)

    # 进行聚类
    clusters1 = dbscan_model.fit_predict(data_processed)
    # 使用PCA降维到2维以便于可视化
    pca = PCA(n_components=5)
    data_2d = pca.fit_transform(data_processed)
    # 绘制散点图
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters1, cmap='viridis')
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    image_data = io.BytesIO()
    plt.savefig(image_data, format='png')
    plt.close()

    # 将图像转换为 Base64 字符串
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

    # labels_true = data_processed
    labels_pred = dbscan_model.fit_predict(data_processed)
    silhouette_score = metrics.silhouette_score(data_processed, labels_pred, metric='euclidean')
    # return "Silhouette Coefficient: %0.3f" % silhouette_score
    # 返回图像的 Base64 字符串
    print("Silhouette Coefficient: %0.3f" % silhouette_score)
    return {'image_base64': image_base64, 'result_text': "Silhouette Coefficient: %0.3f" % silhouette_score}


def km(table_name):
    index = int(table_name[-1])
    query = text(f"SELECT * FROM 'properties' WHERE [index] = {index} ")
    table_data = db.session.execute(query).fetchall()
    column_names = [
        'PropertyID', 'PropType', 'nbhd', 'Style', 'Stories',
        'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms',
        'Fbath', 'Hbath', 'Sale_price', 'RICH', 'index'
    ]
    data = pd.DataFrame(table_data, columns=column_names)
    data = data.drop(columns=[data.columns[-1]])
    non_numeric_cols = ['PropType', 'Style', 'RICH']
    data_encoded = pd.get_dummies(data, columns=non_numeric_cols)
    scaler = StandardScaler()
    numeric_cols = ['nbhd', 'Stories', 'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms', 'Fbath', 'Hbath',
                    'Sale_price']
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols].apply(pd.to_numeric, errors='coerce'))
    data_encoded.dropna(inplace=True)
    data_processed = data_encoded.drop(['PropertyID'], axis=1)

    kmeans_model = KMeans(n_clusters=2)  # 假设我们选择3个簇，这个数字可以根据实际情况调整
    clusters2 = kmeans_model.fit_predict(data_processed)
    # 使用PCA降维到2维以便于可视化
    pca = PCA(n_components=10)
    data_2d = pca.fit_transform(data_processed)
    # 绘制散点图
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters2, cmap='viridis')
    plt.title('KMeans Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    image_data = io.BytesIO()
    plt.savefig(image_data, format='png')
    plt.close()

    # 将图像转换为 Base64 字符串
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

    # 返回图像的 Base64 字符串


    labels_pred = kmeans_model.fit_predict(data_processed)
    silhouette_score = metrics.silhouette_score(data_processed ,labels_pred, metric='euclidean')
    # return "Silhouette Coefficient: %0.3f" % silhouette_score
    return {'image_base64': image_base64, 'result_text': "Silhouette Coefficient: %0.3f" % silhouette_score}

def kme(table_name):
    index = int(table_name[-1])
    query = text(f"SELECT * FROM 'properties' WHERE [index] = {index} ")
    table_data = db.session.execute(query).fetchall()
    column_names = [
        'PropertyID', 'PropType', 'nbhd', 'Style', 'Stories',
        'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms',
        'Fbath', 'Hbath', 'Sale_price', 'RICH', 'index'
    ]
    data = pd.DataFrame(table_data, columns=column_names)
    data = data.drop(columns=[data.columns[-1]])

    non_numeric_cols = ['PropType', 'Style', 'RICH']
    data_encoded = pd.get_dummies(data, columns=non_numeric_cols)
    scaler = StandardScaler()
    numeric_cols = ['nbhd', 'Stories', 'Year_Built', 'Rooms', 'FinishedSqft', 'Units', 'Bdrms', 'Fbath', 'Hbath',
                    'Sale_price']
    data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols].apply(pd.to_numeric, errors='coerce'))
    data_encoded.dropna(inplace=True)
    data_processed = data_encoded.drop(['PropertyID'], axis=1)

    kmedoids_model = KMedoids(n_clusters= auto)  # 假设我们选择3个簇，这个数字可以根据实际情况调整
    clusters3 = kmedoids_model.fit_predict(data_processed)

    # 使用PCA降维到2维以便于可视化
    pca = PCA(n_components=5)
    data_2d = pca.fit_transform(data_processed)

    # 绘制散点图
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=clusters3, cmap='viridis')
    plt.title('KMedoids Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    image_data = io.BytesIO()
    plt.savefig(image_data, format='png')
    plt.close()

    # 将图像转换为 Base64 字符串
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

    # 返回图像的 Base64 字符串


    labels_pred = kmedoids_model.fit_predict(data_processed)
    silhouette_score = metrics.silhouette_score(data_processed ,labels_pred, metric='euclidean')
    # return "Silhouette Coefficient: %0.3f" % silhouette_score
    return {'image_base64': image_base64, 'result_text': "Silhouette Coefficient: %0.3f" % silhouette_score}