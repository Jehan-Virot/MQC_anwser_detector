import pandas as pd
import numpy as np 
import skimage as ski
from functions.gradient_based_filter import gradient_based_filter
import matplotlib.pyplot as plt

image = ski.io.imread("data/FORM1/EXAM_FORM1_PRESENCES/EXAM_FORM1_75012.jpg")
image = np.mean(image, axis=2)
img_edge = gradient_based_filter(image)
img_edge = img_edge - img_edge.min()
img_edge = img_edge / img_edge.max()
img_edge = (img_edge * 255).astype(np.uint8)
ski.io.imsave("edge.jpg",img_edge)