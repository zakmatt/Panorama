#!/usr/bin/env python3
from controllers.controllers import homogeneous,  transform_image, open_image
import cv2
import numpy as np

DEBUG = True

class Panorama(object):
    def __init__(self, image_1, image_2, homography, inverse_homography):
        self.image_1 = image_1
        self.image_2 = image_2
        self.homography = homography
        self.inverse_homography = inverse_homography
        
    def calculate_size(self):
        (h1, w1) = self.image_1.shape[:2]
        (h2, w2) = self.image_2.shape[:2]
      
        #remap the coordinates of the projected image onto the panorama image space
        top_left = np.dot(self.inverse_homography,np.asarray([0,0,1]))
        top_right = np.dot(self.inverse_homography,np.asarray([w2,0,1]))
        bottom_left = np.dot(self.inverse_homography,np.asarray([0,h2,1]))
        bottom_right = np.dot(self.inverse_homography,np.asarray([w2,h2,1]))
    
        if DEBUG:
            print(top_left)
            print(top_right)
            print(bottom_left)
            print(bottom_right)
      
        #normalize
        top_left = top_left/top_left[2]
        top_right = top_right/top_right[2]
        bottom_left = bottom_left/bottom_left[2]
        bottom_right = bottom_right/bottom_right[2]
        
        if DEBUG:
            print(np.int32(top_left))
            print(np.int32(top_right))
            print(np.int32(bottom_left))
            print(np.int32(bottom_right))
      
        pano_left = int(min(top_left[0], bottom_left[0], 0))
        pano_right = int(max(top_right[0], bottom_right[0], w1))
        W = pano_right - pano_left
        
        pano_top = int(min(top_left[1], top_right[1], 0))
        pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
        H = pano_bottom - pano_top
        
        size = (W, H)
        
        if DEBUG:
            print('Panodimensions')
            print(pano_top)
            print(pano_bottom)
            
        # offset of first image relative to panorama
        X = int(min(top_left[0], bottom_left[0], 0))
        Y = int(min(top_left[1], top_right[1], 0))
        offset = (-X, -Y)
      
        if DEBUG:
            print('Calculated size:')
            print(size)
            print('Calculated offset:')
            print(offset)
          
        ## Update the homography to shift by the offset
        # does offset need to be remapped to old coord space?
        # print homography
        # homography[0:2,2] += offset
                    
        self.size = size
        self.offset = offset
  
    
    def stitch(self):
      (h1, w1) = self.image_1.shape[:2]
      (h2, w2) = self.image_2.shape[:2]
      
      panorama = np.zeros((self.size[1], self.size[0], 3), np.uint8)
      
      (ox, oy) = self.offset
      
      translation = np.matrix([
        [1.0, 0.0, ox],
        [0, 1.0, oy],
        [0.0, 0.0, 1.0]
      ])
      
      if DEBUG:
        print(self.inverse_homography)
      self.inverse_homography = translation * self.inverse_homography
      # print homography
      
      # draw the transformed image2
      panorama = cv2.warpPerspective(self.image_2, self.inverse_homography, self.size, panorama)
      panorama[oy:h1+oy, ox:ox+w1] = self.image_1
      cv2.imwrite('result.jpg', panorama)
              
      return panorama

if __name__ == '__main__':
    image_1 = open_image('../images/Rainier1.png')
    image_2 = open_image('../images/Rainier2.png')
    homography = np.array([[  8.36735103e-01,  -1.90076800e-01,  -7.29322463e+01],
                             [  5.97834678e-02,   6.01682560e-01,   5.80613020e+01],
                             [  2.04860178e-04,  -8.05026431e-04,   1.00000000e+00]])
    inverse_homography = np.linalg.inv(homography)
    panorama = Panorama(image_1, image_2, homography, inverse_homography)
    panorama.calculate_size()
    panorama.stitch()