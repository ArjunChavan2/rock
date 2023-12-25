from detection import *
from PIL import Image

path1 = "img1.jpg"
path2 = "img1_annotated.jpg"
image = cv2.imread(path1)
mask = filter_image(image)
keypoints = detect_blob(mask)

blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS())

cv2.imshow("displaying blobs", blobs)