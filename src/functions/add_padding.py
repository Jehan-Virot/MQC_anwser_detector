import numpy as np

def add_zeros_padding(kernel_size:tuple,image):
    padding_x = (kernel_size[0]-1)//2
    padding_y = (kernel_size[1]-1)//2
    image_size_x = image.shape[0]
    image_size_y = image.shape[1]
    try:
        new_image = np.zeros((padding_x*2 + image_size_x, padding_y*2 + image_size_y, image.shape[2]))
        new_image[padding_x:image_size_x+padding_x, padding_y:image_size_y+padding_y, :image.shape[2]] = image
    except:
        new_image = np.zeros((padding_x*2 + image_size_x, padding_y*2 + image_size_y))
        new_image[padding_x:image_size_x+padding_x, padding_y:image_size_y+padding_y] = image

    return new_image

