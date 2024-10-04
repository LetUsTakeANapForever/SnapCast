import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
                imgOut[i+windowSize//2][j+windowSize//2] = 0
                detected_area += 1
            else:
                imgOut[i + windowSize // 2][j + windowSize // 2] = 255
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
    # q_img = quantize(img)
    # q_img_t = quantize(texture)
    img_overlay = textureOverlay(q_img, q_img_t, 0.2)
    resized_img = resize_img(img_overlay[0])
    area = img_overlay[1]
    cir = get_circularity(q_img)
    # print('area:', area)
    # print('circularity', cir)
    return [resized_img, area, cir]


def quantize(img):
    image = Image.open(img)
    quantized_image = image.quantize(colors=4, method='mediancut')
    quantized_image.save('quantized_image.png')


def get_his(img):
        quantize(img)
        qt_img = cv2.imread('quantized_image.png')
        h, w = qt_img.shape[:2]
        color_his = np.full(256, 0, dtype=int)
        for i in range(h):
            for j in range(w):
                color_his[img[i][j][0]] += 1
        return color_his


def detect_cloud_types(img):
    area = cv2.countNonZero(img)
    cir = get_circularity(img)
    print(area)
    print(cir)
    if area < 20000 and 100 < cir < 1000:
        print('Type:cumulus (0)')
    elif area > 20000000 and 100 < cir < 500:
        print('Type:nimbostratus (1)')
    else:
        print('Type:stratocumulus (0)')


def load_file():
    pass


def write_excel():
    pass


cumulus = cv2.imread('cloud_dataset/cumulus.jpg', 0)
nimbostratus = cv2.imread('cloud_dataset/nimbostratus.png', 0)
stratocumulus = cv2.imread('cloud_dataset/stratocumulus.jpg', 0)

cumulus_t = cv2.imread('cloud_dataset/cumulus_t.png', 0)
nimbostratus_t = cv2.imread('cloud_dataset/nimbo_t.png', 0)
stratocumulus_t = cv2.imread('cloud_dataset/strato_t.png', 0)

run1 = run(cumulus, cumulus_t)
run2 = run(nimbostratus, nimbostratus_t)
run3 = run(stratocumulus, stratocumulus_t)

# cv2.imshow('overlay', run1)
# cv2.waitKey(0)
# cv2.imshow('overlay', run2)
# cv2.waitKey(0)
# cv2.imshow('overlay', run3)
# cv2.waitKey(0)

detect_cloud_types(stratocumulus)

