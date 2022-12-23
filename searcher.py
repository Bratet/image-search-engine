# import the necessary packages
import numpy as np
import csv
from scipy.spatial.distance import euclidean
from math import sqrt
class Searcher:

    def __init__(self, indexPath):
        self.indexPath = indexPath
        
    def search(self, queryFeatures, textures = 1, colors = 1, limit = 5):
        results = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = []
                features.append([float(x) for x in row[1:1441]])
                features.append([float(x) for x in row[1441:]])

                d = self.chi2_distance(features[0], queryFeatures[0])

                e = euclidean(features[1], queryFeatures[1])
                normalized_color_similarity = d
                normalized_texture_similarity = e / 250
                normalized_similarity = (normalized_color_similarity * colors + normalized_texture_similarity * textures) / (colors + textures)
                results[row[0]] = normalized_similarity

            f.close()
        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]
        
    def chi2_distance(self, histA, histB, eps = 1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        return d