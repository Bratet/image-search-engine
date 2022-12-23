#Preprocess the imagesin dataset file by resizing them to a uniform size

import cv2
import glob
import os

for imagePath in glob.glob("dataset" + "/*.jpg"):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (220, 220))
    cv2.imwrite(imagePath, image)
