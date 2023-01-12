# import the necessary packages
import numpy as np
import cv2
import imutils
import mahotas
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

class ColorDescriptor:

    def __init__(self, bins):
        self.bins = bins

    def extract_colors_with_HSV(self, image):
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

        features = [float(x) for x in features]
        return features

    def extract_colors_with_RGB(self, image):
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

        features = [float(x) for x in features]
        return features
    
    def extract_colors_with_mean(self, image):
        moyenne = np.mean(image, axis=(0,1))
        moyenne = [float(x) for x in moyenne]
        return moyenne
    
    

    def histogram(self, image, mask):
    
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
            [0, 180, 0, 256, 0, 256])
        
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        
        else:
            hist = cv2.normalize(hist, hist).flatten()

        hist = [float(x) for x in hist]
        return hist

class TextureDescriptor:
    def __init__(self):
        pass

    def extract_textures_with_lbp(self, image):
        #using LBP algorithm to extract textures
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(image, 24, 8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, 24 + 3),
            range=(0, 24 + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features = [float(x) for x in hist.ravel()]
        return features

    def extract_textures_with_glcm(self, image):
        #using GLCM algorithm to extract textures
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(image, [5], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        ASM = graycoprops(glcm, 'ASM')
        # convert float 32 to float

        features = [float(x) for x in contrast.ravel()]
        features.extend([float(x) for x in dissimilarity.ravel()])
        features.extend([float(x) for x in homogeneity.ravel()])
        features.extend([float(x) for x in energy.ravel()])
        features.extend([float(x) for x in correlation.ravel()])
        features.extend([float(x) for x in ASM.ravel()])
        return features
       

    def extract_textures_with_haralick(self, image):
        #using Haralick algorithm to extract textures
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        textures = mahotas.features.haralick(image).mean(axis=0)
        textures = [float(x) for x in textures]
        return textures
       

class ShapeDescriptor:
    def __init__(self):
        pass

    def extract_shape_with_hu(self, image):
        #using Hu moments to extract shapes
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(image,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        huMoments = cv2.HuMoments(M)
        features = [float(x) for x in huMoments.ravel()]
        return features
    
    def extract_shape_with_zernike(self, image):
        #using Zernike moments to extract shapes
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 10
        value = mahotas.features.zernike_moments(image, radius)
        features = [float(x) for x in value]
        return features
    