"""Read the five exam-condition fields on page 1 (graphic / low level).

Each condition has a YES box and a NO box; the selected one is printed as a
solid (filled) square.  Two conditions additionally carry a "Max number"
mini-grid with two cells; when present the field value is the number of the
selected cell, otherwise it is 1 for YES and 0 for NO.

We measure the dark-pixel fraction (ink) inside each candidate box and pick the
darker one.  Geometry is calibrated for the FORM-3 layout and expressed in
normalised fiducial coordinates, so it is robust to scan shift/scale.
"""

import numpy as np

# condition column centres (u) on page 1, left -> right
_CENTERS = [0.101, 0.314, 0.522, 0.736, 0.947]
_YES_DU, _NO_DU = -0.052, 0.042          # box offset from centre
_HALF = 0.017                            # half box width / height (u and v)
_V0, _V1 = 0.616, 0.648                  # YES/NO row (v range)

# conditions carrying a "Max number" mini-grid:
#   index -> (left_val, right_val, left_u, right_u)
_MAX_GRID = {
    1: (0, 1, 0.317, 0.344),
    4: (0, 2, 0.924, 0.950),
}
_MAX_V0, _MAX_V1 = 0.655, 0.693
_MAX_HALF = 0.013

# field labels in page-01 order (rows 5..9)
LABELS = ["Notes de cours", "Notes manuscrites", "Ordinateur portable",
          "Calculatrice ", "Feuilles brouillon"]


def _ink(frame, u, v0, v1):
    sub, _ = frame.roi(u - _HALF, v0, u + _HALF, v1)
    if sub.size == 0:
        return 0.0
    return float((sub < 0.45).mean())


def _shade(frame, u, v0, v1):
    """Fraction of mid-grey background pixels (detects a shaded cell)."""
    sub, _ = frame.roi(u - _MAX_HALF, v0, u + _MAX_HALF, v1)
    if sub.size == 0:
        return 0.0
    return float(((sub > 0.45) & (sub < 0.86)).mean())


def read_conditions(frame):
    """Return a dict label -> int value for the five conditions."""
    out = {}
    for i, center in enumerate(_CENTERS):
        yes_ink = _ink(frame, center + _YES_DU, _V0, _V1)
        no_ink = _ink(frame, center + _NO_DU, _V0, _V1)
        yes_selected = yes_ink > no_ink

        if i in _MAX_GRID:
            lo_val, hi_val, lo_u, hi_u = _MAX_GRID[i]
            lo = _shade(frame, lo_u, _MAX_V0, _MAX_V1)
            hi = _shade(frame, hi_u, _MAX_V0, _MAX_V1)
            value = hi_val if hi >= lo else lo_val
            if not yes_selected:
                value = 0
        else:
            value = 1 if yes_selected else 0

        out[LABELS[i]] = value
    return out
