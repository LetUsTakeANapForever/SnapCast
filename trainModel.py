import cv2
import pandas as pd
from sklearn.cluster import KMeans
import joblib


df = pd.read_csv('cloud_features.csv')
data = df[['Area', 'Circularity', 'Aspect Ratio', 'Edge Density', 'Mean Intensity', 'FEdgeness']]
df.dropna(inplace=True)
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(data)

cluster_labels = kmeans_model.labels_

df['cloud_cluster'] = cluster_labels
print(df.tail(50))

joblib.dump(kmeans_model, 'kmeans_cloud_model.pkl')
