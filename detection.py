import cv2
import numpy as np


grey_lower = np.array([50,50,50])
grey_upper = np.array([200,200,200])

def filter_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, grey_lower, grey_upper)
    return mask

def detect_blob(mask):
    img = cv2.medianBlur(mask, 5)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 400
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return keypoints