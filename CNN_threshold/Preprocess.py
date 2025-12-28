# Preprocess.py

import cv2
import numpy as np
import math

#module level variables
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    #imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY) nen dung he mau HSV
    
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) #lam noi bat bien so hon de tach khoi nen
    #cv2.imwrite("imgGrayscalePlusTopHatMinusBlackHat.jpg",imgMaxContrastGrayscale)
    
    height, width = imgGrayscale.shape
    
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    #cv2.imwrite("gauss.jpg",imgBlurred)
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0
    
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    
    
    #tao anh nhi phan
    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    #mau sac, do bao hoa, gia tri cuong do sang
    #khong chon mau rbg vi anh mau do se con lan cac mau khac kho xac dinh mot mau
    return imgValue

def maximizeContrast(imgGrayscale):
    #lam cho do tuong phan lon nhat
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tao bo loc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)#noi bat chi tiet sang trong nen toi
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)#noi bat chi tiet toi trong nen sang
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat








