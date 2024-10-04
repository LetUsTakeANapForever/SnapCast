import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    FEdgeness = calFEdgeness(img, 100)
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
    for i in range(r-windowSize+1):
        for j in range(c-windowSize+1):
            subImg = img[i:i+windowSize, j:j+windowSize]
            hisSubImg = calTextureHis(subImg)
            if calL1Dist(hisTexture, hisSubImg) < threshold:
                imgOut[i+windowSize//2][j+windowSize//2] = 255
                # change what to do here
    return imgOut


def quantize_img_color(img):
    pass


def detect_edge(img):
    return cv2.Canny(img, threshold1=)

def get_circularity():
    pass


cummulus = cv2.imread('cloud_dataset/cumulus/cumulus1.png',0)
