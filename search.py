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


limit = 5

#######################################################################################

# load the query image and describe it
query = cv2.imread(args["query"])
query = cv2.resize(query, (220, 220))

data = "data.json"

all_methods = {
	"colors" : ['hsv', 'rgb', 'mean'],
	"textures" : ['lbp', 'glcm', 'haralick'],
	"shape" : ['zernike', 'hu']
}


list_of_methods = {
	"colors" : 'hsv',
	"textures" : 'lbp',
	"shapes" : 'zernike'
}

# perform the search
searcher = Searcher(data)
# results = searcher.search_by_color(query, method = "hsv")
# results = searcher.search_by_texture(query, "haralick")
# results = searcher.search_by_shape(query, "hu")
results = searcher.hybrid_search(query, list_of_methods, textures = 1, colors = 1, shapes = 1, limit = 5)

########################################################################################

l = []
# loop over the results
for (score, resultID) in results[:limit]:
	# load the result image and display it
	if resultID < 10:
		result = cv2.imread('dataset\cat_000' + str(resultID) + '.jpg')
	elif resultID < 100:
		result = cv2.imread('dataset\cat_00' + str(resultID) + '.jpg')
	else:
		result = cv2.imread('dataset\cat_0' + str(resultID) + '.jpg')
	l += [cv2.resize(result, (220, 220))]

# show the query
cv2.imshow("Query", query)
# show the result
cv2.imshow("Result", np.concatenate(l, axis=1))
cv2.waitKey(0)