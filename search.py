# import the necessary packages
from descriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
args = vars(ap.parse_args())


# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))


# load the query image and describe it
query = cv2.imread(args["query"])
features = cd.describe(query)


index = "index.csv"


# perform the search
searcher = Searcher(index)
results = searcher.search(features)


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