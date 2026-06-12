"""Page registration using the four corner fiducial marks.

Only low-level operations are used (grayscale, Otsu threshold, connected-
component labelling, affine warp), in line with the project constraints for
graphical elements.

The four L-shaped marks printed at the page corners define a stable reference
rectangle. We detect their centres, then express every region of interest in
normalised coordinates (u, v) in [0, 1] relative to that rectangle:

    u = 0 -> left fiducial column,   u = 1 -> right fiducial column
    v = 0 -> top fiducial row,       v = 1 -> bottom fiducial row

A region of interest (ROI) is therefore independent of scan shift, scale and
small rotations.
"""

import numpy as np
from skimage import color, filters, measure, transform


def to_gray(img):
    """Return a float grayscale image in [0, 1]."""
    if img.ndim == 3:
        return color.rgb2gray(img[..., :3])
    img = img.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _corner_fiducial(binv, y_slice, x_slice):
    """Largest compact dark blob inside a corner quadrant -> (row, col) centre."""
    sub = binv[y_slice, x_slice]
    labels = measure.label(sub)
    best = None
    for p in measure.regionprops(labels):
        if p.area < 60:
            continue
        minr, minc, maxr, maxc = p.bbox
        h, w = maxr - minr, maxc - minc
        # fiducials are small square-ish brackets; reject text/lines
        if not (10 < h < 160 and 10 < w < 160):
            continue
        if best is None or p.area > best.area:
            best = p
    if best is None:
        return None
    minr, minc, maxr, maxc = best.bbox
    cy = (minr + maxr) / 2.0 + y_slice.start
    cx = (minc + maxc) / 2.0 + x_slice.start
    return np.array([cy, cx], dtype=float)


def detect_fiducials(gray):
    """Detect the four corner fiducials.

    Returns a dict with keys TL, TR, BL, BR, each an array [row, col], or None
    if any corner is missing.
    """
    binv = gray < filters.threshold_otsu(gray)
    h, w = gray.shape
    mx, my = int(0.18 * w), int(0.14 * h)
    corners = {
        "TL": _corner_fiducial(binv, slice(0, my), slice(0, mx)),
        "TR": _corner_fiducial(binv, slice(0, my), slice(w - mx, w)),
        "BL": _corner_fiducial(binv, slice(h - my, h), slice(0, mx)),
        "BR": _corner_fiducial(binv, slice(h - my, h), slice(w - mx, w)),
    }
    if any(v is None for v in corners.values()):
        return None
    return corners


class PageFrame:
    """Maps normalised (u, v) coordinates to pixel coordinates of one page."""

    def __init__(self, gray, fiducials):
        self.gray = gray
        self.f = fiducials
        # column (x) reference from top/bottom fiducial pairs
        self.x_left = 0.5 * (fiducials["TL"][1] + fiducials["BL"][1])
        self.x_right = 0.5 * (fiducials["TR"][1] + fiducials["BR"][1])
        self.y_top = 0.5 * (fiducials["TL"][0] + fiducials["TR"][0])
        self.y_bot = 0.5 * (fiducials["BL"][0] + fiducials["BR"][0])
        self.w = self.x_right - self.x_left
        self.h = self.y_bot - self.y_top

    def px(self, u, v):
        """Normalised (u, v) -> (row, col) pixel."""
        return (self.y_top + v * self.h, self.x_left + u * self.w)

    def roi(self, u0, v0, u1, v1):
        """Crop a rectangular ROI given in normalised coordinates."""
        r0 = int(round(self.y_top + v0 * self.h))
        r1 = int(round(self.y_top + v1 * self.h))
        c0 = int(round(self.x_left + u0 * self.w))
        c1 = int(round(self.x_left + u1 * self.w))
        r0, r1 = sorted((max(0, r0), max(0, r1)))
        c0, c1 = sorted((max(0, c0), max(0, c1)))
        return self.gray[r0:r1, c0:c1], (r0, c0, r1, c1)


def build_frame(img):
    """Convenience: image -> (PageFrame or None, gray)."""
    gray = to_gray(img)
    fids = detect_fiducials(gray)
    if fids is None:
        return None, gray
    return PageFrame(gray, fids), gray
