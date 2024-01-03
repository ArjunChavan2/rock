import cv2
import numpy as np

fileNum = 2
path1 = f"data\\img{fileNum}.jpg" #Input file
path2 = f"annotated\\img{fileNum}_annotated_mask_yellow.jpg" #Output file
img = cv2.imread(path1)


if img is None:
    print("\n\n----------\n\nError: Image not loaded successfully\n\n----------\n\n")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Converting image to HSV color
    
    grey_lower = np.array([0, 0, 40]) #Lower HSV range for grey
    grey_upper = np.array([180, 10, 180]) #Upper HSV range for grey

    color_lower = np.array([20,100,100]) #Lower HSV range for a specified chromatic color
    color_upper = np.array([30,255,255]) #Upper HSV range for a specified chromatic color
 
    """
    Yellow HSV Range:
    color_lower = np.array([20,100,100])
    color_upper = np.array([30,255,255])

    Green HSV Range:
    color_lower = np.array([40,40,40])
    color_upper = np.array([80,255,255])

    Orange HSV Range:
    color_lower = np.array([0,100,100])
    color_upper = np.array([20,255,255])
    """

    grey_mask = cv2.inRange(hsv, grey_lower, grey_upper) #Creates a mask that excludes any chromatic color(non grey)
    color_mask = cv2.inRange(hsv, color_lower, color_upper) #Creates a mask that excludes anything not of the specified color

    non_grey_mask = cv2.bitwise_not(grey_mask) #Takes the opposite of the grey mask, so everything not grey

    mask = cv2.bitwise_and(color_mask, color_mask, mask=non_grey_mask) #Combining the grey and colored masks
    #cv2.imwrite("annotated\\img2_mask_yellow.jpg", mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Applying the masks and finding white "countours"
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < 10000000: #Checks if area of the contour(hold) found is within 300 and 10000000 pixels
            filtered_contours.append(contour) #If the area fits, then it is added to a list of holds.

    result_img = img.copy()
    cv2.drawContours(mask, filtered_contours, -1, (0, 0, 255), 2) #Adds an outline to the holds found onto a copy of the original image
    cv2.imwrite(path2, mask)











