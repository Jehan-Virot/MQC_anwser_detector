"""Generic optical-mark grid reader.

Used for the STUDENT ID grid (5 columns x 10 digits) and the GROUP grid.
Only low-level operations are used: histogram stretch, Otsu threshold, simple
morphology, connected-component labelling, 1-D coordinate clustering and a
regularity search to snap detected square cells onto a clean lattice.  No
high-level "find checkbox" primitive is used.
"""

import itertools
import numpy as np
from skimage import exposure, filters, measure, morphology


def stretch(g, p_low=5, p_high=95):
    lo, hi = np.percentile(g, (p_low, p_high))
    if hi <= lo:
        return g
    return exposure.rescale_intensity(g, in_range=(lo, hi), out_range=(0, 1))


def binarize(g, otsu_scale=0.95):
    gs = stretch(g)
    t = filters.threshold_otsu(gs) * otsu_scale
    b = gs < t
    b = morphology.remove_small_objects(b, min_size=7)
    b = morphology.closing(b, morphology.footprint_rectangle((2,2)))
    return b


def _square_candidates(b):
    """Connected components that look like empty/checked cell outlines."""
    labels = measure.label(b, connectivity=2)
    h, w = b.shape
    cands = []
    for p in measure.regionprops(labels):
        minr, minc, maxr, maxc = p.bbox
        ch, cw = maxr - minr, maxc - minc
        if ch <= 0 or cw <= 0:
            continue
        min_side = max(8, int(0.04 * h))
        max_side = max(20, int(0.28 * h))
        ratio = cw / ch
        extent = p.area / float(cw * ch)
        if min_side <= cw <= max_side and min_side <= ch <= max_side:
            if 0.6 <= ratio <= 1.7 and 0.05 <= extent <= 0.9:
                cands.append({"cx": (minc + maxc) / 2.0,
                              "cy": (minr + maxr) / 2.0,
                              "w": cw, "h": ch})
    return cands


def _cluster(values, tol):
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [{"center": float(np.mean(g)), "count": len(g)} for g in groups]


def _regular(clusters, n):
    """Pick n clusters whose centres are as evenly spaced as possible."""
    if len(clusters) < n:
        return None
    clusters = sorted(clusters, key=lambda c: c["center"])
    best, best_cost = None, float("inf")
    max_count = max(c["count"] for c in clusters)
    for combo in itertools.combinations(clusters, n):
        centers = np.array([c["center"] for c in combo])
        gaps = np.diff(centers)
        if np.any(gaps <= 0):
            continue
        mean_gap = gaps.mean()
        cost = gaps.std() / mean_gap - 0.10 * sum(
            c["count"] for c in combo) / (n * max_count)
        if cost < best_cost:
            best_cost, best = cost, combo
    if best is None:
        return None
    return sorted(c["center"] for c in best)


def reconstruct_grid(b, n_cols, n_rows):
    """Return (col_centers, row_centers, side) or (None, None, None)."""
    cands = _square_candidates(b)
    if len(cands) < max(8, int(0.3 * n_cols * n_rows)):
        return None, None, None
    sides = [0.5 * (c["w"] + c["h"]) for c in cands]
    side = float(np.median(sides))
    tol = max(5, 0.6 * side)
    xs = _cluster([c["cx"] for c in cands], tol)
    ys = _cluster([c["cy"] for c in cands], tol)
    col_c = _regular(xs, n_cols)
    row_c = _regular(ys, n_rows)
    if col_c is None or row_c is None:
        return None, None, None
    return col_c, row_c, side * 1.05


def cell_fill(b, cy, cx, side, pad=0.27):
    """Ink fraction inside a cell, ignoring its printed border."""
    h, w = b.shape
    half = side / 2.0
    p = int(pad * side)
    r0 = max(0, int(cy - half) + p)
    r1 = min(h, int(cy + half) - p)
    c0 = max(0, int(cx - half) + p)
    c1 = min(w, int(cx + half) - p)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return 0.0
    inner = b[r0:r1, c0:c1]
    return inner.sum() / inner.size


def read_digit_grid(gray_roi, n_cols, n_rows, min_fill=0.030, otsu_scale=0.92):
    """Read a column-major digit grid.

    Each column encodes one digit (the marked row index 0..n_rows-1).
    Returns dict with 'value' (string or None), per-column fills and 'status'.
    """
    b = binarize(gray_roi, otsu_scale)
    col_c, row_c, side = reconstruct_grid(b, n_cols, n_rows)
    if col_c is None:
        return {"value": None, "status": "grid_not_found", "fills": None,
                "grid": None}
    fills = np.zeros((n_rows, n_cols))
    for ci, cx in enumerate(col_c):
        for ri, cy in enumerate(row_c):
            fills[ri, ci] = cell_fill(b, cy, cx, side)
    digits, ambiguous = [], False
    for ci in range(n_cols):
        col = fills[:, ci]
        r = int(np.argmax(col))
        if col[r] < min_fill:
            ambiguous = True
            digits.append("?")
        else:
            digits.append(str(r))
    value = None if ambiguous else "".join(digits)
    return {"value": value,
            "status": "ok" if value else "weak_marks",
            "fills": fills,
            "grid": (col_c, row_c, side)}


def read_group(gray_roi, otsu_scale=0.92, min_fill=0.030):
    """Read the GROUP block: 2 digit columns (0-9) + 1 letter column (A-J).

    Robust to the printed header boxes and the A-J glyph labels: we keep only
    x-clusters that contain roughly a full column of cells, take the three
    right-most as the checkbox columns, and read 10 evenly spaced rows.
    """
    b = binarize(gray_roi, otsu_scale)
    cands = _square_candidates(b)
    if len(cands) < 12:
        return {"value": None, "status": "grid_not_found"}
    side = float(np.median([0.5 * (c["w"] + c["h"]) for c in cands]))
    xcl = [c for c in _cluster([c["cx"] for c in cands], max(6, 0.6 * side))
           if c["count"] >= 6]
    if len(xcl) < 3:
        return {"value": None, "status": "columns_not_found"}
    cols = sorted(c["center"] for c in xcl)[-3:]  # 3 right-most real columns
    cys = sorted(c["cy"] for c in cands)
    y0, y1 = np.percentile(cys, 5), np.percentile(cys, 95)
    rows = np.linspace(y0, y1, 10)
    out = []
    for cx in cols:
        fills = [cell_fill(b, cy, cx, side) for cy in rows]
        r = int(np.argmax(fills))
        out.append(r if fills[r] >= min_fill else None)
    d1, d2, lt = out
    if None in out:
        return {"value": None, "status": "weak_marks", "raw": out}
    code = f"{d1}{d2}{chr(ord('A') + lt)}"
    return {"value": code, "status": "ok", "raw": out}
