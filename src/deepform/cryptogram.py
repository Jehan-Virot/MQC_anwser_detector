"""Cryptogram extraction and cross-page consistency.

The cryptogram is the small graphic printed at the bottom of every page.  It
must be identical on all pages; otherwise pages may have been swapped between
students.  We extract it from a fixed region below the fiducial box, normalise
it to a small binary bitmap and compare every page against page 1 with a
simple normalised correlation (low-level template matching).
"""

import numpy as np
from skimage import filters, transform

# normalised ROI (relative to the fiducial box) of the bottom cryptogram
CRYPTO_ROI = (0.125, 1.000, 0.225, 1.055)
BITMAP_SIZE = 48
MATCH_THRESHOLD = 0.80  # min agreement fraction to consider pages identical


def extract_cryptogram(frame):
    """Return a normalised binary bitmap of the cryptogram, or None."""
    roi, _ = frame.roi(*CRYPTO_ROI)
    if roi.size == 0:
        return None
    roi = np.clip(roi, 0, 1)
    try:
        t = filters.threshold_otsu(roi)
    except ValueError:
        return None
    b = (roi < t).astype(np.float32)
    if b.sum() < 5:  # essentially empty -> nothing printed
        return None
    bm = transform.resize(b, (BITMAP_SIZE, BITMAP_SIZE),
                          preserve_range=True, anti_aliasing=True)
    return (bm > 0.5).astype(np.uint8)


def agreement(a, b):
    """Fraction of pixels that match between two binary bitmaps."""
    if a is None or b is None:
        return 0.0
    return float(np.mean(a == b))


def check_consistency(bitmaps):
    """Given per-page cryptogram bitmaps, return (is_consistent, min_score).

    Pages with no detectable cryptogram are ignored.
    """
    valid = [b for b in bitmaps if b is not None]
    if len(valid) < 2:
        return (len(valid) == 1), 1.0 if valid else 0.0
    ref = valid[0]
    scores = [agreement(ref, b) for b in valid[1:]]
    worst = min(scores)
    return worst >= MATCH_THRESHOLD, worst
