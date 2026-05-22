from skimage import io, color, transform, exposure, filters
from numpy import percentile

def load_binary_image(path):
    try:
        img = io.imread(path)
        
        img_gray = rgb_to_grey_and_or_normalisation(img)
        etire_img_gray = strech_histo(img_gray)
        
        treshold = filters.threshold_otsu(etire_img_gray)
        img_binaire = etire_img_gray < treshold
        img_binaire = turn_image(img_binaire)
        
        return img_binaire
    except:
        return None

def rgb_to_grey_and_or_normalisation(img):
    if img.ndim == 3:
        return color.rgb2gray(img)
    else:
        return img / 255.0

def turn_image(img):
    H, W = img.shape
    if W > H:
        img = transform.rotate(img, angle=270, resize=True)
    return img

def strech_histo(img):
    low, high = percentile(img,(5,95))
    etire = exposure.rescale_intensity(img, in_range=(low, high), out_range=(0,1))
    return etire
