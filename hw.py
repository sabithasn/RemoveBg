
import cv2
import time
import numpy as np

# Dimensions of the files to be read to resize to same size.
dim = (183,249)
file_name = "dog.jpg"
img = cv2.imread(file_name, 1)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
bg_file = "bg.jpg"
bg = cv2.imread(bg_file, 1)
bg= cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)
#Converting the color from BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Generating mask to detect black colour
#These values can also be changed as per the color
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
mask_1 = cv2.inRange(hsv, lower_black, upper_black)

#Open and expand the image where there is mask 1 (color)
mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

#Selecting only the part that does not have mask one and saving in mask 2
mask_2 = cv2.bitwise_not(mask_1)

#Keeping only the part of the images without the black color 
#(or any other color you may choose)
res_1 = cv2.bitwise_and(img, img, mask=mask_2)

#Keeping only the part of the images with the black color
#(or any other color you may choose)
res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

#Generating the final output by merging res_1 and res_2
final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
cv2.imwrite("travel.png", final_output)





