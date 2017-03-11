#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:10:08 2017

@author: Matthew Zak
"""

from controllers.controllers import *
import cv2
from controllers.harris import Harris
from controllers.feature_description import *
from controllers.feature_matching import *
from controllers.panorama import Panorama
from controllers.ransac import Ransac
import numpy as np


def get_matching_corners(image_1, image_2):
    description_1, corners_1, key_points_1 = get_image_description(
            image_1, 5, 1, 0.3
            )
    description_2, corners_2, key_points_2 = get_image_description(
            image_2, 5, 1, 0.3
            )
    
    matches = match_features(description_1, 
                             description_2, 
                             0.8)
    
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]
    
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'normal_match.jpg')
    
    matching_indices = [(match.queryIdx, match.trainIdx) for 
                        match in matches]
    image_1_corners = [key_point.pt for key_point in key_points_1]
    image_2_corners = [key_point.pt for key_point in key_points_2]
    matching_corners_1 = np.array([list(image_1_corners[pair[0]]) for pair in matching_indices], dtype = np.int)
    matching_corners_2 = np.array([list(image_2_corners[pair[1]]) for pair in matching_indices], dtype = np.int)
    matching_corners = np.array([[a[0], a[1], b[0], b[1]] for a, b in zip(matching_corners_1, matching_corners_2)])
    
    return matching_corners

def draw_matches_after_ransac(image_1, image_2, best_points):
    key_points_1 = [cv2.KeyPoint(point[0], point[1], 15) for point in best_points[:, :2]]
    key_points_2 = [cv2.KeyPoint(point[0], point[1], 15) for point in best_points[:, 2:]]
    distances = np.sqrt(((best_points[:, :2] - best_points[:, 2:]) ** 2).sum(1))
    matches = [cv2.DMatch(i, i, distances[i]) for i in range(distances.size)]
    im3 = cv2.drawMatches(image_1, 
                          key_points_1,
                          image_2, 
                          key_points_2, 
                          matches,
                          None,
                          flags = 2)
    save_image(im3, 'ransac_matches.jpg')
    
if __name__ == '__main__':
    # Matching     
    image_1 = open_image('images/Rainier1.png')
    image_2 = open_image('images/Rainier2.png')
    matching_corners = get_matching_corners(image_1, image_2)
    
    ransac = Ransac(matching_corners, 0.5, 100)
    ransac.run()
    best_indices = ransac.best_indices.flatten()
    best_points = matching_corners[best_indices]
    draw_matches_after_ransac(image_1, image_2, best_points)
    ransac.find_best_homography()
    homography = ransac.best_homography
    inverse_homography = ransac.inverse_homography
    panorama = Panorama(image_1, image_2, homography, inverse_homography)
    panorama.calculate_size()
    panorama.stitch()