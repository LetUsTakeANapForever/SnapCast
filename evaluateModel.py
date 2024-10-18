from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import pandas as pd


kmeans_model = joblib.load('kmeans_cloud_model.pkl')

df = pd.read_csv('cloud_features.csv')
X_test = df[['Area', 'Circularity', 'Aspect Ratio', 'Edge Density', 'Mean Intensity', 'FEdgeness']]

cluster_labels = kmeans_model.predict(X_test)

df_class = pd.read_csv('cloud_class.csv')
y_true = df['Class']


cloud_classes = ['cumulus', 'nimbo', 'strato']
y_true = [cloud_classes.index(label) for label in y_true]

print('Confusion Matrix:')
print(confusion_matrix(y_true, cluster_labels))

print('Accuracy:')
print(accuracy_score(y_true, cluster_labels))
