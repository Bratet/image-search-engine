# import the necessary packages
from descriptor import ColorDescriptor, TextureDescriptor
import glob
import cv2

# initialize the descriptor
cd = ColorDescriptor((8, 12, 3))
td = TextureDescriptor()



with open("indexx.csv", "w") as output:
	for imagePath in glob.glob("dataset" + "/*.jpg"):

		imageID = imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)

		features = cd.describe(image)
		features.extend(td.extract_textures(image))

		features = [str(f) for f in features]
		output.write("%s,%s\n" % (imageID, ",".join(features)))
