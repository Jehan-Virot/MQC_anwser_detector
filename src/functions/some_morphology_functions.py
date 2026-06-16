from scipy import ndimage as ndi
import numpy as np

def count_8_neighbors(binary_img):
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=np.uint8
    )

    return ndi.convolve(binary_img.astype(np.uint8),kernel,mode="constant",cval=0)
