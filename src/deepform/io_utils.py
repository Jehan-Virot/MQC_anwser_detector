"""Input/output helpers: PDF -> page images, image loading."""

import numpy as np
import fitz  # PyMuPDF
from skimage import io as skio


def pdf_to_page_images(pdf_path, dpi=150):
    """Return a list of RGB uint8 page images (one per PDF page)."""
    doc = fitz.open(pdf_path)
    pages = []
    for pg in doc:
        pix = pg.get_pixmap(dpi=dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # drop alpha
            arr = arr[..., :3]
        elif pix.n == 1:  # grayscale -> rgb
            arr = np.repeat(arr, 3, axis=2)
        pages.append(np.ascontiguousarray(arr[..., :3]))
    doc.close()
    return pages


def load_image(path):
    return skio.imread(path)
