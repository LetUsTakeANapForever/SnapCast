import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('cloud_features.csv')
data = df[['Area', 'Centroid X', 'Centroid Y', 'ExternalCornerCount', 'InternalCornerCount', 'Perimeter',
           'Circularity']]
df.dropna(inplace=True)
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(data)

cluster_labels = kmeans_model.labels_

df['cluster'] = cluster_labels
print(df.head(50))
