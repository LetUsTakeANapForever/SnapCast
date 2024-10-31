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


# def quantize_img_color(img):
#     r, c = img.shape
#     quantized_image = img.copy()
#     for i in range(r):
#         for j in range(c):
#             if 0 <= img[i, j] <= 63:
#                 quantized_image[i, j] = [0, 0, 0]
#             elif 64 <= img[i, j] <= 127:
#                 quantized_image[i, j] = [100, 100, 100]
#             elif 128 <= img[i, j] <= 191:
#                 quantized_image[i, j] = [150, 150, 150]
#             else:
#                 quantized_image[i, j] = [255, 255, 255]
#     return quantized_image


def get_circularity(img):
    edge = cv2.Canny(img, threshold1=100, threshold2=200)
    perimeter = cv2.countNonZero(edge)
    area = cv2.countNonZero(img)
    return round(perimeter**2/area)


def resize_img(img):
    h, w = img.shape
    return cv2.resize(img, (w//2, h//2))


# def run(img, texture):
#     q_img = quantize_img_color(img)
#     q_img_t = quantize_img_color(texture)
#     img_overlay = textureOverlay(q_img, q_img_t, 0.2)
#     resized_img = resize_img(img_overlay[0])
#     return resized_img


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
    if (area > 50000 and cir < 1000 and aspect_ratio > 1.4 and 0.02 <= edge_density <= 0.05
            and 40 < mean_intensity < 181):
        print('Type:cumulus (0)')
    elif (area > 50000 and cir < 250 and aspect_ratio < 1.51 and edge_density < 0.02
          and mean_intensity < 230):
        print('Type:nimbostratus (1)')
    elif (area > 80000 and cir > 1000 and aspect_ratio > 1.3 and 0.03 < edge_density < 0.08
          and mean_intensity > 69):
        print('Type:stratocumulus (0)')


cumulus = cv2.imread('bigcumu21.jpg', 0)
nimbostratus = cv2.imread('bignimbo7.jpg', 0)
stratocumulus = cv2.imread('bigstrato16.jpg', 0)


resized_cumulus = resize_img(cumulus)
resized_nimbostratus = resize_img(nimbostratus)
resized_stratocumulus = resize_img(stratocumulus)

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
detect_cloud_types(cumulus)
# print('\n')
# detect_cloud_types(nimbostratus)
# print('\n')
# detect_cloud_types(stratocumulus)
