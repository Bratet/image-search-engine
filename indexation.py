# import the necessary packages
from descriptor import ColorDescriptor, TextureDescriptor, ShapeDescriptor
import glob
import cv2
import json

# initialize the descriptor
cd = ColorDescriptor((8, 12, 3))
td = TextureDescriptor()
sh = ShapeDescriptor()

path = glob.glob("dataset" + "/*.jpg")
index = 0
l = len(path)

with open("data.json", "w") as output:
    json_file = []
    for imagePath in path:
        image = cv2.imread(imagePath)

        features = {}
        # extract the color features by HSV
        hsv = cd.extract_colors_with_HSV(image)
        features["hsv"] = hsv
        # extract the color features by RGB
        rgb = cd.extract_colors_with_RGB(image)
        features["rgb"] = rgb
        # extract color features by mean
        mean = cd.extract_colors_with_mean(image)
        features["mean"] = mean

        # extract the texture features by lbp
        lbp = td.extract_textures_with_lbp(image)
        features["lbp"] = lbp
        # extract the texture features by glcm
        glcm = td.extract_textures_with_glcm(image)
        features["glcm"] = glcm
        # extract the texture features by haralick
        haralick = td.extract_textures_with_haralick(image)
        features["haralick"] = haralick

        # exrtact the shape features by zernike
        zernike = sh.extract_shape_with_zernike(image)
        features["zernike"] = zernike
        # extract the shape features by hu
        hu = sh.extract_shape_with_hu(image)
        features["hu"] = hu

        # save the features in json file
        json_file.append(features)

        index += 1

        progress = "{:.2f}".format((index / l)*100)
        print(f"{progress}%")


        
    json.dump(json_file, output)