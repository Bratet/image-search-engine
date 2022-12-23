# import the necessary packages
import numpy as np
import cv2
import imutils
import skimage.feature as feature

class ColorDescriptor:

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
    
        for (startX, endX, startY, endY) in segments:
        
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
        
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
    
        return features
    
    def histogram(self, image, mask):
    
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
            [0, 180, 0, 256, 0, 256])
        
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        
        else:
            hist = cv2.normalize(hist, hist).flatten()
        
        return hist



class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps = 1e-10):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints+3), range=(0, self.numPoints + 2))

        # Normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        return hist, lbp

class TextureDescriptor:
    def __init__(self):
        pass

    def extract_textures(self, image):
        #using LocalBinaryPatterns algorithm to extract textures
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = LocalBinaryPatterns(24, 8)
        (hist, lbp) = lbp.describe(image)
        return lbp.ravel()
       