import cv2
import numpy as np

fileNum = 5
path1 = f"data\\img{fileNum}.jpg"
path2 = f"annotated\\img{fileNum}_annotated.jpg"
img = cv2.imread(path1)

class Hold:
    def __init__(self, keypoint, points):
        self.keypoint = keypoint
        self.points = points

if img is None:
    print("\n\n----------\n\nError: Image not loaded successfully\n\n----------\n\n")
else:
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray_blur = cv2.GaussianBlur(imgray, (3, 3), 0)
    thresh, img_bw = cv2.threshold(imgray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(img_bw, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        #IMG 6: 500, 70000
        if 500 < area < 10000:
            filtered_contours.append(contour)

    result_img = img.copy()
    cv2.drawContours(result_img, filtered_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(path2, result_img)
