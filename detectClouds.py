import cv2
import numpy as np
import os
import pandas as pd


def calMeanGray(img):
    return np.average(img) / 255


def calFEdgeness(img, threshold):
    img_edge = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    img_edge = np.abs(img_edge)
    r, c = img_edge.shape[:2]
    FEdgeness = np.count_nonzero(img_edge > threshold) / (r*c)
    return FEdgeness


def calTextureHis(img):
    normalizedMean = calMeanGray(img)
    FEdgeness = calFEdgeness(img, 138)
    textureHis = np.append(normalizedMean, FEdgeness)
    return textureHis


def calL1Dist(his1, his2):
    index0 = his1[0]-his2[0]
    index1 = his1[1]-his2[1]
    l1 = np.abs(index0) + np.abs(index1)
    return l1


def textureOverlay(img, texture, threshold):
    imgOut = np.copy(img)
    hisTexture = calTextureHis(texture)
    windowSize = 21
    r, c = img.shape[:2]
    detected_area = 0
    for i in range(r-windowSize+1):
        for j in range(c-windowSize+1):
            subImg = img[i:i+windowSize, j:j+windowSize]
            hisSubImg = calTextureHis(subImg)
            if calL1Dist(hisTexture, hisSubImg) < threshold:
                imgOut[i+windowSize//2][j+windowSize//2] = 255
                detected_area += 1
            else:
                imgOut[i + windowSize // 2][j + windowSize // 2] = 0
    return [imgOut, detected_area]


def quantize_img_color(img):
    if len(img.shape) == 2:
        data = img.reshape((-1, 1))
    else:
        data = img.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized_data = centers[labels.flatten()]
    quantized_image = quantized_data.reshape(img.shape)
    return quantized_image


def get_circularity(img):
    edge = cv2.Canny(img, threshold1=100, threshold2=200)
    perimeter = cv2.countNonZero(edge)
    area = cv2.countNonZero(img)
    return round(perimeter**2/area)


def resize_img(img):
    h, w = img.shape
    return cv2.resize(img, (w//2, h//2))


def run(img, texture):
    q_img = quantize_img_color(img)
    q_img_t = quantize_img_color(texture)
    img_overlay = textureOverlay(q_img, q_img_t, 0.2)
    resized_img = resize_img(img_overlay[0])
    area = img_overlay[1]
    cir = get_circularity(q_img)
    return [resized_img, area, cir]


def get_edgeDensity(img):
    r, c = img.shape
    smooth_img = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(smooth_img, threshold1=100, threshold2=200)
    total_pixels = r*c
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / total_pixels
    return edge_density


def get_aspect_ratio(img):
    # Get the shape of the image (height, width)
    h, w = img.shape[:2]
    # Calculate aspect ratio
    aspect_ratio = w / h
    return aspect_ratio


def get_mean_intensity(img):
    return np.mean(img)


def resize_to_fixed_size(img, width, height):
    # Resize the image to the specified width and height
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img


def detect_cloud_types(img):
    if 'cumulus' in str(img):
        texture = cumulus_t
    elif 'nimbo' in str(img):
        texture = nimbostratus_t
    else:
        texture = stratocumulus_t

    picture_overlay = textureOverlay(img, texture, 0.2)[0]

    area = cv2.countNonZero(picture_overlay)
    cir = get_circularity(picture_overlay)
    aspect_ratio = get_aspect_ratio(picture_overlay)
    edge_density = get_edgeDensity(picture_overlay)
    mean_intensity = get_mean_intensity(picture_overlay)
    print("Area:", area)
    print("Circularity:", cir)
    print("Aspect Ratio:", aspect_ratio)
    print("Edge Density:", edge_density)
    print("Mean Intensity:", mean_intensity)
    if area > 3300 and 19 <= cir < 120 and 0.8 < aspect_ratio < 3.6 and edge_density > 0.038 and mean_intensity > 100:
        print('Type:cumulus (0)')
    elif (area >= 2800 or 12 <= cir < 52 and 1.1 < aspect_ratio < 2.2 and 0.0 <= edge_density < 0.05
          and mean_intensity < 100):
        print('Type:nimbostratus (1)')
    elif (5000 < area < 55000 or cir >= 99 and 0.8 < aspect_ratio < 1.17 and 0.02 < edge_density < 0.08
          and mean_intensity > 30):
        print('Type:stratocumulus (0)')


def img_properties(img):
    if 'cumulus' in str(img):
        texture = cumulus_t
    elif 'nimbo' in str(img):
        texture = nimbostratus_t
    else:
        texture = stratocumulus_t

    picture_overlay = textureOverlay(img, texture, 0.2)[0]

    area = cv2.countNonZero(picture_overlay)
    cir = get_circularity(picture_overlay)
    aspect_ratio = get_aspect_ratio(picture_overlay)
    edge_density = get_edgeDensity(picture_overlay)
    mean_intensity = get_mean_intensity(picture_overlay)
    return [area, cir, aspect_ratio, edge_density, mean_intensity, calFEdgeness(img, 135)]


def load_sub_folders(folder):
    folder_names = []
    for folder_name in os.listdir(folder):
        if folder_name is not None:
            folder_names.append(folder_name)
    return folder_names


def load_img_in_subfolder(folder):
    file_names = []
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is not None:
            file_names.append(file)
    return file_names


def get_all_images():
    images = []
    for subfolder_name in load_sub_folders('cloud_dataset'):
        for filename in load_img_in_subfolder('cloud_dataset/' + subfolder_name):
            img_path = 'cloud_dataset/' + subfolder_name + '/' + filename
            img = cv2.imread(img_path, 0)
            images.append(img)
    return images


def generate_excel():
    subfolder_names_list = []
    filename_list = []
    area_list = []
    cir_list = []
    aspect_ratio_list = []
    edge_density_list = []
    mean_intensity_list = []
    fEdgeness_list = []

    for subfolder_name in load_sub_folders('cloud_dataset'):
        for filename in load_img_in_subfolder('cloud_dataset/'+subfolder_name):
            subfolder_names_list.append(subfolder_name)
            filename_list.append(filename)

    for img in get_all_images():
        if 'cumulus' in str(img)[0:7]:
            texture = cumulus_t
        elif 'nimbo' in str(img)[0:5]:
            texture = nimbostratus_t
        else:
            texture = stratocumulus_t

        resized_cumulus = resize_to_fixed_size(img, fixed_width, fixed_height)
        picture_overlay = textureOverlay(resized_cumulus, texture, 0.2)[0]
        area_list.append(cv2.countNonZero(picture_overlay))
        cir_list.append(get_circularity(picture_overlay))
        aspect_ratio_list.append(get_aspect_ratio(picture_overlay))
        edge_density_list.append(get_edgeDensity(picture_overlay))
        mean_intensity_list.append(get_mean_intensity(picture_overlay))
        fEdgeness_list.append(calFEdgeness(resized_cumulus, 138))

    data = {
        'Class': subfolder_names_list,
        'Filename': filename_list,
        'Area': area_list,
        'Circularity': cir_list,
        'Aspect Ratio': aspect_ratio_list,
        'Edge Density': edge_density_list,
        'Mean Intensity': mean_intensity_list,
        'FEdgeness': fEdgeness_list
    }

    df = pd.DataFrame(data)
    df.to_csv('cloud_features.csv', index=False)


fixed_width = 256
fixed_height = 256

img_test = cv2.imread('cloud_dataset/nimbo/mininimbo5.png', 0)
# cumulus = cv2.imread('cloud_dataset/cumulus/minicumulus100.png', 0)
# nimbostratus = cv2.imread('cloud_dataset/nimbo/nimbost47.png', 0)
# stratocumulus = cv2.imread('cloud_dataset/strato/miniStrato40.png', 0)

resized_img_test = resize_to_fixed_size(img_test, fixed_width, fixed_height)
# resized_cumulus = resize_to_fixed_size(cumulus, fixed_width, fixed_height)
# resized_nimbostratus = resize_to_fixed_size(nimbostratus, fixed_width, fixed_height)
# resized_stratocumulus = resize_to_fixed_size(stratocumulus, fixed_width, fixed_height)

cumulus_t = cv2.imread('cumulus_t.png', 0)
nimbostratus_t = cv2.imread('nimbo_t.png', 0)
stratocumulus_t = cv2.imread('strato_t.png', 0)

# run1 = run(resized_cumulus, cumulus_t)
# run2 = run(resized_nimbostratus, nimbostratus_t)
# run3 = run(resized_stratocumulus, stratocumulus_t)
#
# cv2.imshow('original', resized_cumulus)
# cv2.waitKey(0)
# cv2.imshow('original', resized_nimbostratus)
# cv2.waitKey(0)
# cv2.imshow('original', resized_stratocumulus)
# cv2.waitKey(0)
#
# cv2.imshow('overlay', run1[0])
# cv2.waitKey(0)
# cv2.imshow('overlay', run2[0])
# cv2.waitKey(0)
# cv2.imshow('overlay', run3[0])
# cv2.waitKey(0)
#
# detect_cloud_types(cumulus)
# print('\n')
# detect_cloud_types(nimbostratus)
# print('\n')
# detect_cloud_types(stratocumulus)

detect_cloud_types(resized_img_test)

# generate_excel()
