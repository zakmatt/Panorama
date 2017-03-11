#!/usr/bin/env python3
from controllers.controllers import homogeneous
import numpy as np

class Panorama(object):
    def __init__(self, image_1, image_2, homography, inverse_homography):
        self.image_1 = image_1
        self.image_2 = image_2
        self.homography = homography
        self.inverse_homography = inverse_homography
    
    def stitch(self, sigma = 50):
        def corners():
            i_h = self.inverse_homography
            img = self.image_2
            h, w = img.shape[0:2]
            corners = np.array([[0, w, w, 0],
                                [0, 0, h, h]], dtype=float)
            corners = homogeneous(corners)
            A = np.dot(i_h, corners)
            A = homogeneous(A)
            A = A.astype(np.int)[:2]
            self.w1 = np.min(A[0])
            self.w2 = np.max(A[0])
            self.h1 = np.min(A[1])
            self.h2 = np.max(A[1])
            
            output_range = np.array([(self.w1, self.h1), (self.w2, self.h2)])
            return output_range
            
        output_range = corners()
        # wrap images
        h, w = self.image_1.shape[0:2]
        yy, xx = np.mgrid[0:h, 0:w]
        dist = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
        gwt = np.exp(-dist / (2.0 * sigma ** 2))
        
        # add the gaussian weight as the 4th channel
        npimg = np.dstack((self.image_1, gwt))
        
        # append the warped image to the list
        warpedImg = transform_image(npimg, self.homography, output_range)