from skimage import measure

def find_regions(img_binaire):
    labels = measure.label(img_binaire, connectivity=2)
    return measure.regionprops(labels)
