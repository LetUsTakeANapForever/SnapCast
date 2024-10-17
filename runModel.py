import cv2
import joblib
import detectClouds as dt
import pandas as pd

img = cv2.imread('cloud_dataset/nimbo/mininimbo5.png', 0)
cv2.imshow('', img)
cv2.waitKey(0)
resized_img = dt.resize_to_fixed_size(img, 256, 256)
properties = dt.img_properties(resized_img)

print(properties)
dt.detect_cloud_types(img)

properties_df = pd.DataFrame([properties], columns=[
    'Area', 'Circularity', 'Aspect Ratio', 'Edge Density', 'Mean Intensity', 'FEdgeness'
])
kmeans_model = joblib.load('kmeans_cloud_model.pkl')
predicted_class = kmeans_model.predict(properties_df)

cloud_classes = ['Cumulus', 'Nimbostratus', 'Stratocumulus']
print(f'Predicted cloud type: {cloud_classes[predicted_class[0]]}')
