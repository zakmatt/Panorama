#!/usr/bin/env python3
from controllers.controllers import project, homography
import cv2
import math
import numpy as np

class Ransac(object):
    def __init__(self, data, inlier_threshold = 0.5, max_iterations = 100):
        self.data = data
        self.inlier_threshold = inlier_threshold
        self.max_iterations = max_iterations
        
    def compute_inlier_count(self, points, homography):
        image_1_points = project(points[:, 0:2], homography)
        image_2_points = points[:, 2:].transpose()
        error = np.sqrt((np.array(image_1_points - image_2_points, dtype = np.float32)**2).sum(0))
    
        inlier_count = (error < self.inlier_threshold).sum()
        return inlier_count, error
    
    def run(self):
        iterations = 0
        best_model = None
        best_count = 0
        best_indices = None
        confidence=0.95
        
        # if we reached the maximum iteration
        while iterations < self.max_iterations:
            # make two copies of the data
            copy_data = np.matrix(np.copy(self.data))
            shuffe_data = np.matrix(np.copy(self.data))
            
            # shuffle data and get random 4 points
            np.random.shuffle(shuffe_data)
            points = np.matrix(shuffe_data)[0:4]
            #build homography
            h = homography(points[:, 0:2], points[:, 2:])
            # h, _ = cv2.findHomography(points[:, 0:2], points[:, 2:])
            inliers_count, error = self.compute_inlier_count(copy_data, h)
            
            # if this homography is better than previous ones keep it
            if inliers_count > best_count:
                best_model = h
                best_count = inliers_count
                best_indices = np.argwhere(error < self.inlier_threshold)

                # recalculate max_iterations
                p = float(inliers_count) / self.data.shape[0]
                self.max_iterations = math.log(1 - confidence) / math.log(1 - (p ** 4))
            
            
            # increment iterations
            iterations += 1
            
        if best_model is None:
            raise ValueError("No good model found.")
        else:
            self.best_model = best_model
            self.best_indices = best_indices
            
    def find_best_homography(self):
        best_indices = self.best_indices.flatten()
        best_points = self.data[best_indices]
        self.best_homography, _  = cv2.findHomography(best_points[:, 0:2], 
                                                      best_points[:, 2:])
        self.inverse_homography = np.linalg.inv(self.best_homography)