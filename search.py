# import the necessary packages
from descriptor import ColorDescriptor, TextureDescriptor
from searcher import Searcher
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query",
	help = "Path to the query image")
ap.add_argument("-c", "--color", 
	help = "color parameter")
ap.add_argument("-t", "--texture",
	help = "texture parameter")
ap.add_argument("-n", "--n",
	help = "number of results")
args = vars(ap.parse_args())


# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))
# initialize the texture descriptor
td = TextureDescriptor()


# load the query image and describe it
query = cv2.imread(args["query"])
query = cv2.resize(query, (220, 220))
features = []
features.append(cd.describe(query))
features.append(td.extract_textures(query))

index = "indexx.csv"

# perform the search
searcher = Searcher(index)
results = searcher.search(features, textures = float(args["texture"]), colors = float(args["color"]), limit = int(args["n"]))

l = []
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
	result = cv2.imread(resultID)
	l += [cv2.resize(result, (220, 220))]

# show the query
cv2.imshow("Query", query)
# show the result
cv2.imshow("Result", np.concatenate(l, axis=1))
cv2.waitKey(0)