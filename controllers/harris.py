#!/usr/bin/env python3
from controllers.controllers import *
import cv2
import numpy as np
from numpy import linalg as LA
from scipy.ndimage import filters

class Harris(object):
    
    def __init__(self, image, window_size, sigma, threshold):
        #h, w, _ = image.shape
        #image = cv2.resize(image, (h//2, w//2))
        self.image = image
        self.window_size = window_size
        self.sigma = sigma
        self.threshold = threshold
    
    def gradient_matrix(self):
        h, w, _ = self.image.shape
        self.gradient_matrix = np.zeros((h, w, 4))
        for y in range(h):
            for x in range(w):
                d_x = float(self.I_x[y, x])
                d_y = float(self.I_y[y, x])
                
                # for the position get length and the angle
                # between derivatives in the range od -pi to pi
                self.gradient_matrix[y, x] = [
                        d_x,
                        d_y,
                        LA.norm([d_x, d_y]),
                        cv2.fastAtan2(d_y, d_x)
                        ]
            
    def harris_matrix_adaptive(self):
        
        def adaptive_non_max_suppression(corner_list, max_value):
            
            def identical(corner_1, corner_2):
                if corner_1[0] != corner_2[0]:
                    return False
                elif corner_1[1] != corner_2[1]:
                    return False
                elif corner_1[2] != corner_2[2]:
                    return False
                else:
                    return True
                    
            def distance(corner_1, corner_2):
                y = np.abs(corner_1[0] - corner_2[0])
                x = np.abs(corner_1[1] - corner_2[1])
                r = np.sqrt(x ** 2 + y ** 2)
                return r
            
            adaptive_suppresion_list = []
            # y
            # x
            # c
            for corner_1 in corner_list:
                y = corner_1[0]
                x = corner_1[1]
                if corner_1[2] == max_value:
                    adaptive_suppresion_list.append([y, x, np.inf])
                    continue
                
                current_radius = np.inf
                for corner_2 in corner_list:
                    if identical(corner_1, corner_2):
                        continue
                    
                    r = distance(corner_1, corner_2)
                    if corner_2[2] > corner_1[2] and r < current_radius:
                        current_radius = r
                        
                adaptive_suppresion_list.append([y, x, current_radius])
                
            adaptive_suppresion_list = sorted(adaptive_suppresion_list, 
                                              key = lambda x:x[2],
                                              reverse = True)
            return np.array(adaptive_suppresion_list)    
        
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # DoG filtering
        kernel_dog = d_o_g(21, 1.5, 1.0)
        gray_image = cv2.filter2D(gray_image, -1, kernel_dog)
        
        self.I_x, self.I_y = compute_derivatives(gray_image)
        I_xx = self.I_x ** 2
        I_yy = self.I_y ** 2
        I_xy = self.I_x * self.I_y
        
        h, w, _ = self.image.shape
        kernel = GaussianKernel(self.window_size, self.sigma)
        corner_list = []
        self.key_points = []
        corners_map = np.zeros((h,w))
        
        I_xx = np.array(cv2.filter2D(I_xx, -1, kernel), dtype = np.float32)
        I_yy = np.array(cv2.filter2D(I_yy, -1, kernel), dtype = np.float32)
        I_xy = np.array(cv2.filter2D(I_xy, -1, kernel), dtype = np.float32)
        
        # compute Harris feature strength, avoiding divide by zero
        harris_matrix = (I_xx * I_yy - I_xy ** 2) / (I_xx + I_yy + 1e-8)
        
        # exclude points near the image border
        harris_matrix[:, :16] = 0
        harris_matrix[:, -16:] = 0
        harris_matrix[:16, :] = 0
        harris_matrix[-16:, :] = 0
            
        max_value = np.max(harris_matrix)
        self.threshold *= max_value
        suppress_pos = harris_matrix < self.threshold
        harris_matrix[suppress_pos] = 0.0
        
        max_y, max_x = np.nonzero(harris_matrix)
        indices = [pos for pos in zip(max_y, max_x)]
        for index in indices:
            y = index[0]
            x = index[1]
            corner_list.append([y, x, harris_matrix[y, x]])
            
        print('get corners')
        adaptive_corners = adaptive_non_max_suppression(corner_list,
                                                        max_value)
        print('corners found')
        
        adaptive_corners = adaptive_corners[:50]
        for corner in adaptive_corners:
            y = corner[0]
            x = corner[1]
            self.key_points.append(cv2.KeyPoint(x, y, 15))
            
        self.corner_list = adaptive_corners
        self.corners_map = corners_map
        output_image = cv2.drawKeypoints(rescale(self.image).astype('uint8'), self.key_points, self.image)
        self.output_image = output_image

if __name__ == '__main__':
    image = open_image('Yosemite1.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris = Harris(image, 5, 1, 0.3)
    harris.harris_matrix_adaptive()