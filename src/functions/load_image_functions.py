from skimage import io, color


def load_binary_image(path, dark_threshold):
    try:
        img = io.imread(path)

        gray = rgb_to_grey_and_or_normalisation(img)

        binary = gray < dark_threshold

        return binary
    except:
        return None

def rgb_to_grey_and_or_normalisation(img):
    if img.ndim == 3:
        return color.rgb2gray(img)
    else:
        return img / 255.0


