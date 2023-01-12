# import the necessary packages
import numpy as np
from math import sqrt
import json
from descriptor import ColorDescriptor, TextureDescriptor, ShapeDescriptor

# chi-squared distance of two images
def chi2_distance(histA, histB, eps = 1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        return d


def centre_reduce_list(list):


    # calculate the mean of the population
    mean = np.mean(list, axis=0)
    # calculate the variance of the population
    variance = np.var(list, axis=0)
    # calculate the standard deviation of the population
    std = (sqrt(variance[0]), 
           sqrt(variance[1]),
            sqrt(variance[2])
            )

    # calculate the centre reduced list
    centre_reduced_list = []
    for i in range(len(list)):
        centre_reduced_list.append(((list[i][0] - mean[0])/std[0],
                                      (list[i][1] - mean[1])/std[1],
                                        (list[i][2] - mean[2])/std[2],
                                        i))

    return centre_reduced_list


class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath
    
    def search_by_color(self, query, method, limit = 5):
        results = []

        with open(self.indexPath) as f:
            # read json data
            data = json.load(f)
            try:
                cd = ColorDescriptor((8, 12, 3))
                if method == "hsv":
                    queryFeatures = cd.extract_colors_with_HSV(query)
                elif method == "rgb":
                    queryFeatures = cd.extract_colors_with_RGB(query)
                elif method == "mean":
                    queryFeatures = cd.extract_colors_with_mean(query)
                else:
                    raise("Invalid method")

                for i in range(len(data)):
                    d = chi2_distance(data[i][method], queryFeatures)
                    results.append(d)
                return sorted([(v, k) for (k, v) in enumerate(results)])[:limit]
            except:
                print("Invalid method")
                return None
        
    def search_by_texture(self, query, method, limit = 5):
        results = []

        with open(self.indexPath) as f:
            # read json data
            data = json.load(f)
            try:
                td = TextureDescriptor()
                if method == "lbp":
                    queryFeatures = td.extract_textures_with_lbp(query)
                elif method == "glcm":
                    queryFeatures = td.extract_textures_with_glcm(query)
                elif method == "haralick":
                    queryFeatures = td.extract_textures_with_haralick(query)
                else:
                    raise("Invalid method")

                for i in range(len(data)):
                    d = chi2_distance(data[i][method], queryFeatures)
                    results.append(d)
                return sorted([(v, k) for (k, v) in enumerate(results)])[:limit]
            except Exception as e:
                print(str(e))
                return None
    
    def search_by_shape(self, query, method, limit = 5):
        results = []

        with open(self.indexPath) as f:
            # read json data
            data = json.load(f)
            try:
                sd = ShapeDescriptor()
                if method == "zernike":
                    queryFeatures = sd.extract_shape_with_zernike(query)
                elif method == "hu":
                    queryFeatures = sd.extract_shape_with_hu(query)
                else:
                    raise("Invalid method")

                for i in range(len(data)):
                    d = chi2_distance(data[i][method], queryFeatures)
                    results.append(d)
                return sorted([(v, k) for (k, v) in enumerate(results)])[:limit]
            except Exception as e:
                print(str(e))
                return None



    def hybrid_search(self, query, list_of_methods, textures = 1, colors = 1, shapes = 1, limit = 5):
        results = []
        with open(self.indexPath) as f:
            # read json data
            data = json.load(f)
            try:
                cd = ColorDescriptor((8, 12, 3))
                td = TextureDescriptor()
                sd = ShapeDescriptor()
                queryFeatures = {}
                if list_of_methods['colors'] == "hsv":
                    queryFeatures['colors'] = cd.extract_colors_with_HSV(query)
                elif list_of_methods['colors'] == "rgb":
                    queryFeatures['colors'] = cd.extract_colors_with_RGB(query)
                elif list_of_methods['colors'] == "mean":
                    queryFeatures['colors'] = cd.extract_colors_with_mean(query)
                else:
                    raise("Invalid colors method")
                
                if list_of_methods['textures'] == "lbp":
                    queryFeatures['textures'] = td.extract_textures_with_lbp(query)
                elif list_of_methods['textures'] == "glcm":
                    queryFeatures['textures'] = td.extract_textures_with_glcm(query)
                elif list_of_methods['textures'] == "haralick":
                    queryFeatures['textures'] = td.extract_textures_with_haralick(query)
                else:
                    raise("Invalid textures method")
                
                if list_of_methods['shapes'] == "zernike":
                    queryFeatures['shapes'] = sd.extract_shape_with_zernike(query)
                elif list_of_methods['shapes'] == "hu":
                    queryFeatures['shapes'] = sd.extract_shape_with_hu(query)
                else:
                    raise("Invalid shapes method")
                
                for i in range(len(data)):
                    results.append((chi2_distance(data[i][list_of_methods['colors']], queryFeatures['colors']), 
                                    chi2_distance(data[i][list_of_methods['textures']], queryFeatures['textures']),
                                    chi2_distance(data[i][list_of_methods['shapes']], queryFeatures['shapes']),
                                    i))
                
                reduced_features = centre_reduce_list(results)

                for i in range(len(reduced_features)):
                    reduced_features[i] = ((reduced_features[i][0]*colors + reduced_features[i][1]*textures + reduced_features[i][2]*shapes)/(colors + textures + shapes), reduced_features[i][3])
                

                return sorted(reduced_features, key=lambda x: x[0])[:limit]

                
            
            except Exception as e:
                print(str(e))
                return None
