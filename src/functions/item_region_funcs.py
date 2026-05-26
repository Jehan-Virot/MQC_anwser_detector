import numpy as np
from skimage import measure

def remove_small_objects(img_binaire, min_size=40, connectivity=2):
    labels = measure.label(img_binaire, connectivity=connectivity)
    output = np.zeros_like(img_binaire, dtype=bool)

    for region in measure.regionprops(labels):
        if region.area >= min_size:
            output[labels == region.label] = True

    return output