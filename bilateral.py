import cv2
import matplotlib 
import numpy as np
import random
import sys
import math
from math import log
'''
I = cv2.imread('test2.hdr', flags = cv2.IMREAD_ANYDEPTH ).astype(np.float32)
B = cv2.bilateralFilter(I, 5, 10, 1)
cv2.imwrite( 'bilateral_cv.jpg', B)
'''

def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever img_in is not a 2D float32 valued np.ndarray
    """

    # check the input
    if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D np.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum



# function definition here....

# read the lena image, convert to float and scale to [0,1]
I = cv2.imread('pic_station.hdr', flags = cv2.IMREAD_ANYDEPTH ).astype(np.float32)
print(I.shape)
np.savetxt('I.csv', I[:,:,0], delimiter=',', fmt='%f')
# bilateral filter the image
B = np.stack([ 
        filter_bilateral( I[:,:,0], 10.0, 1 ),
        filter_bilateral( I[:,:,1], 10.0, 1 ),
        filter_bilateral( I[:,:,2], 10.0, 1 )], axis=2 )

np.savetxt('B.csv', B[:,:,0], delimiter=',', fmt='%f')
# stack the images horizontally
O = np.hstack( [I,B] )
print(O.shape)
# write out the image
cv2.imwrite( 'bilateral_pic_station.jpg', O)


