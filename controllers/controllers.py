#!/usr/bin/env python3
import cv2
from math import exp
import numpy as np
from scipy.ndimage import interpolation

def GaussianKernel(size, sigma):
    centre = size // 2 + 1
    
    def get_coeffs(pos_x, pos_y):
        return exp(-1.0 * ((pos_x - centre)**2 +
                           (pos_y - centre)**2) / (2 * sigma**2))
    
    gaussian_filter = np.zeros((size, size))
    for pos_x in range(size):
        for pos_y in range(size):
            gaussian_filter[pos_x, pos_y] = get_coeffs(pos_x+1, pos_y+1)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

# DoG - Difference of Gaussians
def d_o_g(k_size, sigma_1, sigma_2):
    kernel_1 = GaussianKernel(k_size, sigma_1)
    kernel_2 = GaussianKernel(k_size, sigma_2)
    return kernel_1 - kernel_2

def rescale(image):
    image = image.astype('float32')
    current_min = np.min(image)
    current_max = np.max(image)
    image = (image - current_min)/(current_max - current_min) * 255
    return image

def compute_derivatives(image):
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    return (sobelx, sobely)

def color_image(image, indices):
    colored_image = image.copy()
    for position in indices:
        y, x = position
        colored_image[y, x, 0] = 0
        colored_image[y, x, 1] = 0
        colored_image[y, x, 2] = 255
                     
    return colored_image

def open_image(image_path):
    image = cv2.imread(image_path)
    image = np.array(image, dtype = np.float32)
    return image

def save_image(image, path):
    image = rescale(image)
    image = np.array(image, dtype = np.uint8)
    cv2.imwrite(path, image)
    
def homography(p1, p2):
    """Finds the homography that maps points from p1 to p2
    @p1: a Nx2 array of positions that correspond to p2, N >= 4
    @p2: a Nx2 array of positions that correspond to p1, N >= 4
    @return: a 3x3 matrix that maps the points from p1 to p2
    p2=Hp1
    """
       
    # check if there is at least 4 points
    if p1.shape[0] < 4 or p2.shape[0] < 4:
       raise ValueError("p1 and p2 must have at least 4 row")
           
    # create matrix A
    A = np.zeros((p1.shape[0] * 2, 8), dtype=float)
    A = np.matrix(A, dtype=float)

    # fill A
    for i in range(0, A.shape[0]):
        # if i is event
        if i % 2 == 0:
            A[i, 0] = p1[i / 2, 0]
            A[i, 1] = p1[i / 2, 1]
            A[i, 2] = 1
            A[i, 6] = -p2[i / 2, 0] * p1[i / 2, 0]
            A[i, 7] = -p2[i / 2, 0] * p1[i / 2, 1]
        # if i is odd
        else:
            A[i, 3] = p1[i / 2, 0]
            A[i, 4] = p1[i / 2, 1]
            A[i, 5] = 1
            A[i, 6] = -p2[i / 2, 1] * p1[i / 2, 0]
            A[i, 7] = -p2[i / 2, 1] * p1[i / 2, 1]

    # create vector b
    b = p2.flatten()
    # b = b.reshape(b.shape[1], 1)
    b = b.reshape(b.shape[1], 1)
    b = b.astype(float)
    
    # calculate homography Ax=b
    if p1.shape[0] == 4:
        x = np.linalg.solve(A, b)
    else:
        x = np.linalg.lstsq(A, b)[0]
    # reshape x
    x = np.vstack((x, np.matrix(1)))
    x = x.reshape((3, 3))

    return x    

def homogeneous(xyw_in):
    """
    Converts a 2xN or 3xN set of points to homogeneous coordinates.
        - for 2xN arrays, adds a row of ones
        - for 3xN arrays, divides all rows by the third row
    """
    if xyw_in.shape[0] == 3:
        xywOut = np.zeros_like(xyw_in)
        for i in range(3):
            xywOut[i, :] = xyw_in[i, :] / xyw_in[2, :]
    elif xyw_in.shape[0] == 2:
        xywOut = np.vstack((xyw_in, np.ones((1, xyw_in.shape[1]), dtype=xyw_in.dtype)))
    else:
        raise ValueError("xyw_in is not 2xN or 3xN")

    return xywOut    

def project(points, homography):
    points = points.transpose()
    points = homogeneous(points)
    
    # compute transformed points
    transformed_points = (homography * points)
    transformed_points = np.array(homogeneous(transformed_points)[0:2, :], dtype = np.int)
    return transformed_points

def transform_image(image, transform, output_range):
    h, w, _ = image.shape
    transform = np.array(transform)
    
    # determine pixel coordinates of an output image
    w1 = output_range[0, 0]
    h1 = output_range[0, 1]
    w2 = output_range[1, 0]
    h2 = output_range[1, 1]
    
    # set up the new window size
    yy, xx = np.mgrid[h1:h2, w1:w2]
    h = h2 - h1
    w = w2 - w1
    
    # transform output pixel coordinates into input image coordinates
    xywOut = np.vstack((xx.flatten(), yy.flatten()))
    xywOut = homogeneous(xywOut)
    xywIn = np.dot(np.linalg.inv(transform), xywOut)
    xywIn = homogeneous(xywIn)
    
    # reshape input image coordinates
    xxIn = xywIn[0, :].reshape((h, w))
    yyIn = xywIn[1, :].reshape((h, w))
    
    print('dim: %d', image.ndim)
    if image.ndim == 3:
        output = np.zeros((h, w, image.shape[2]), dtype = image.dtype)
        for dimension in range(image.shape[2]):
            output[..., dimension] = interpolation.map_coordinates(
                    image[..., dimension],
                    [yyIn, xxIn]
                  )
    return output