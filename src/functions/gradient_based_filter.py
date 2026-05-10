import numpy as np
from functions.add_padding import add_zeros_padding
from math import sqrt

def gradient_based_filter(image):
    sobel_kernel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_kernel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel_size = (3,3)
    no_padding_x = image.shape[0]
    no_padding_y = image.shape[1]
    image = add_zeros_padding(kernel_size, image)
    im_grad_delta = np.zeros((no_padding_x, no_padding_y), dtype=float)
    for x in range(1,no_padding_x-kernel_size[0]):
        for y in range(1, no_padding_y-kernel_size[1]):
            gx = np.sum(image[x-1:x+2,y-1:y+2] * sobel_kernel_x)
            gy = np.sum(image[x-1:x+2,y-1:y+2] * sobel_kernel_y)
            im_grad_delta[x][y] = sqrt(((gx)**2 + (gy)**2))

    return im_grad_delta


