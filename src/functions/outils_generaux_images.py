from skimage import color, transform, exposure, io, filters, morphology
import numpy as np
import random


#######################################
#random

def set_seed(seed=5626):
    random.seed(seed)
    np.random.seed(seed)
    
    
#######################################
#image loading / basic preprocessing

def load_gray_image(path):
    img = io.imread(path)
    img_gray = rgb_to_gray_and_normalize(img)
    img_gray = turn_image_if_needed(img_gray)
    return img_gray


def rgb_to_gray_and_normalize(img):
    if img.ndim == 3:
        return color.rgb2gray(img)
    img = img.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def turn_image_if_needed(img):
    H, W = img.shape[:2]
    if W > H:
        img = transform.rotate(img, angle=270, resize=True, preserve_range=True)
    return img


def stretch_histo(img_gray, p_low=5, p_high=95):
    low, high = np.percentile(img_gray, (p_low, p_high))
    if high <= low:
        return img_gray
    return exposure.rescale_intensity(img_gray, in_range=(low, high), out_range=(0, 1))

#version complete avec image->grayscale->histrogram stretching->otsu
def load_binary_image(path):
    img_gray = load_gray_image(path)
    img_stretched = stretch_histo(img_gray)
    threshold = filters.threshold_otsu(img_stretched)
    img_binaire = img_stretched < threshold
    return img_binaire

##########################################################
#outils de calcules sécurisées 

def safe_div(a, b, eps=1e-8):
    return float(a) / float(b + eps)

##########################################################
#stretching + otsu pour les signature boxes
def binarize_dark_pixels_for_id(img_gray, threshold_factor=0.92):
    img_stretched = stretch_histo(img_gray, p_low=3, p_high=97)
    otsu = filters.threshold_otsu(img_stretched)
    threshold = otsu * threshold_factor

    img_bin = img_stretched < threshold
    img_bin = morphology.remove_small_objects(img_bin, max_size=8)
    img_bin = morphology.closing(img_bin, morphology.footprint_rectangle((2,2)))
    return img_bin

