from skimage import io, color, transform, exposure, filters
from numpy import percentile

def load_binary_image(path, is_not_signature = True):
    try:
        img = io.imread(path)
        img_gray = rgb_to_grey_and_or_normalisation(img)
        
        if is_not_signature:
            img_gray = strech_histo(img_gray)
        
        treshold = filters.threshold_otsu(img_gray)    
        img_binaire = img_gray < treshold
        
        if is_not_signature:
            img_binaire = turn_image(img_binaire)      
        
        if not is_not_signature and (img_binaire.shape[0] != 342 or img_binaire.shape[1] != 239):
            img_binaire = transform.resize(img_binaire, (342, 239))  

        return img_binaire
    except:
        print(f"[ERREUR] {path} -> {type(e).__name__}: {e}")
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
