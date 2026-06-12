"""OCR of printed fields (page 1 header band and the grade boxes).

Printed text is allowed to use higher-level tools per the project rules, so we
use Tesseract here.  Values are parsed by anchoring on the known printed
labels of the header band, which makes parsing robust to spacing noise.
"""

import re
import numpy as np
import pytesseract


def _ocr(gray_roi, config="--psm 7"):
    if gray_roi.size == 0:
        return ""
    img = (np.clip(gray_roi, 0, 1) * 255).astype(np.uint8)
    return pytesseract.image_to_string(img, config=config).strip()


def _clean_module(s):
    # e.g. "1S.2703" -> "IS.2703"
    s = s.strip().rstrip(".")
    s = re.sub(r"^1S", "IS", s)
    return s


def _clean_code(s):
    # e.g. "$2-02-G2" -> "S2-02-G2"
    s = s.strip().replace("$", "S").replace(" ", "")
    return s


def parse_header_band(frame):
    """Read Module / Professor / Date / Code from the top header band.

    Returns a dict with those four keys (values may be '' on failure).
    """
    from skimage import transform
    band, _ = frame.roi(0.02, 0.045, 0.98, 0.085)
    if band.size == 0:
        return {"Module": "", "Professor": "", "Date": "", "Code": ""}
    band = transform.rescale(band, 2.0, anti_aliasing=True)  # help small print
    txt = _ocr(band, config="--psm 7")
    out = {"Module": "", "Professor": "", "Date": "", "Code": ""}

    # Date: global match anywhere in the line
    m = re.search(r"\d{2}/\d{2}/\d{4}", txt)
    if m:
        out["Date"] = m.group(0)

    # split the line on the known labels, keeping order
    norm = txt.replace("odule", "Module").replace("|", " ")
    pattern = r"(Module|Professor|Date|Code)\b[:\s]*"
    parts = re.split(pattern, norm)
    for i in range(1, len(parts) - 1, 2):
        lab, val = parts[i], parts[i + 1].strip()
        if lab in out and not out[lab]:
            out[lab] = val

    # Professor: keep only the leading alphabetic token (drop trailing date/noise)
    if out["Professor"]:
        mp = re.match(r"[A-Za-zÀ-ÿ'’\-]+", out["Professor"].strip())
        if mp:
            out["Professor"] = mp.group(0).strip().upper()
    out["Module"] = _clean_module(out["Module"])
    out["Code"] = _clean_code(out["Code"])
    return out


def read_int_box(frame, roi_norm):
    """Read a single printed integer from a boxed region (e.g. note maximale).

    The digit is small and sits inside a bordered grey box, so we drop the
    border, upscale, binarise and pad before OCR.
    """
    from skimage import filters, transform
    sub, _ = frame.roi(*roi_norm)
    if sub.size == 0:
        return None
    h, w = sub.shape
    sub = sub[int(0.18 * h):int(0.82 * h), int(0.12 * w):int(0.88 * w)]
    if sub.size == 0:
        return None
    sub = transform.rescale(sub, 3, anti_aliasing=True)
    try:
        t = filters.threshold_otsu(sub)
    except ValueError:
        return None
    b = ((sub >= t) * 255).astype(np.uint8)        # black digit on white
    b = np.pad(b, 30, constant_values=255)
    for psm in ("7", "8", "6", "10", "13"):
        txt = pytesseract.image_to_string(
            b, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
        m = re.search(r"\d+", txt)
        if m:
            return int(m.group(0))
    return None
