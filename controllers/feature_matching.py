#!/usr/bin/env python3
import argparse
from controllers.controllers import *
import cv2
from controllers.harris import Harris
from controllers.feature_description import get_description
import numpy as np

def get_image_description(image, window_size, sigma, threshold):
    harris = Harris(image, window_size, sigma, threshold)
    harris.harris_matrix_adaptive()
    harris.gradient_matrix()
    description = get_description(harris)
    return description, harris.corner_list, harris.key_points

def histogram_to_vector(feature_histogram):
    feature_histogram = feature_histogram.flatten()
    feature_histogram = feature_histogram / (np.sqrt(np.sum(feature_histogram ** 2)) + 1e-8)
    bright_points = feature_histogram > 0.2
    feature_histogram[bright_points] = 0.2
    return feature_histogram

def match_features(features_1, features_2, ratio_threshold):
    
    def feature_distance(feature_1, feature_2):
        feature_1 = histogram_to_vector(feature_1)
        feature_2 = histogram_to_vector(feature_2)
        distance = ((feature_1 - feature_2) ** 2).sum()

        return distance
    
    features_matches = []
    matches = []
    flag = False
    # second option - create matrix. rows features_1, columns features_2
    # subtract horizontal (f_2) from vertical (f_1) and get a matrix
    # pick lowest points
    for pos_feat_1, feature_1 in enumerate(features_1):
        best_match = np.inf
        second_best_match = np.inf
        for pos_feat_2, feature_2 in enumerate(features_2):
            
            distance = feature_distance(feature_1, feature_2)
            
            if distance < best_match:
                second_best_match = best_match
                best_match = distance
                flag = True
            # what if distance is equal second_best_match
            # and the computet ratio is also above
            # the threshold? It gives a good matching point
            elif distance < second_best_match:
                second_best_match = distance
                flag = True
                
                
            if flag:
                ratio = 1.0 * best_match / second_best_match
                flag = False
                if distance < 0.3 and ratio < ratio_threshold:
                    features_matches.append(
                            {'feature_a': pos_feat_1,
                             'feature_b': pos_feat_2,
                             'distance': distance
                             }
                            )
                    matches.append(
                            cv2.DMatch(pos_feat_1, pos_feat_2, distance)
                            )
    return matches

def print_points(feature_matches, corners_a, corners_b):
    # each element is [x, y, c] - coordinates and corner strength
    corners_a = [(element[0], element[1]) for element in corners_a]
    corners_b = [(element[0], element[1]) for element in corners_b]
    for features in feature_matches:
        feature_a = features['feature_a']
        feature_b = features['feature_b']
        print('positions: ', feature_a, feature_b)
        print('coordinate of feature_a : (%d, %d)' % (corners_a[feature_a]))
        print('coordinate of feature_b : (%d, %d)' % (corners_b[feature_b]))
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_1",
                        "--image_1",
                        help="First image to match",
                        required=True)
    parser.add_argument("-i_2",
                        "--image_2",
                        help="Second image to match",
                        required=True)
    parser.add_argument("-w",
                        "--window_size",
                        help="Window size",
                        required=True)
    parser.add_argument("-s",
                        "--sigma",
                        help="Sigma value",
                        required=True)
    parser.add_argument("-t",
                        "--threshold",
                        help="Threshold value",
                        required=True)
    parser.add_argument("-r_t",
                        "--ratio_threshold",
                        help="Ratio threshold",
                        required=True)
    args = parser.parse_args()
    image_1 = args.image_1
    image_2 = args.image_2
    window_size = int(args.window_size)
    sigma = int(args.sigma)
    threshold = float(args.threshold)
    ratio_threshold = float(args.ratio_threshold)
    
    image_1 = open_image(image_1)
    image_2 = open_image(image_2)
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, window_size, sigma, threshold
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, window_size, sigma, threshold
            )
    
    feature_matches, matches = match_features(description_1, 
                                     description_2, 
                                     ratio_threshold)
    
    #print(feature_matches.shape)
    feature_matches = sorted(feature_matches,key = lambda x:x['distance'])
    feature_matches = feature_matches[:10]
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    cv2.imwrite('matched_result.jpg', im3)