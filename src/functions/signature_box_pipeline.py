from functions.labels import find_regions

def find_signature_box(item_regions, image_shape):
    H, W = image_shape
    signature_box_possibilities = []

    for p in item_regions:
        minr, minc, maxr, maxc = p.bbox
        h = maxr - minr
        w = maxc - minc

        is_trop_large = w > 0.20 * W and h > 0.08 * H
        is_pas_assez_large = w < 0.50 * W and h < 0.25 * H
        ratio_correct = h > 0 and 1.2 < (w / h) < 3.5

        if is_trop_large and is_pas_assez_large and ratio_correct:
            signature_box_possibilities.append((minc, minr, maxc, maxr))

    if len(signature_box_possibilities) == 0:
        return None

    # On garde la dernière possibilité comme dans ton code.
    x1, y1, x2, y2 = signature_box_possibilities[-1]
    return x1, y1, x2, y2


def detect_if_the_signature_is_there(img_binaire, signature_box, ratio_pixel=0.006):
    if signature_box is None:
        return False, 0.0

    x1, y1, x2, y2 = signature_box
    w = x2 - x1
    h = y2 - y1

    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    inside = img_binaire[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
    if inside.size == 0:
        return False, 0.0

    ink_ratio = inside.sum() / inside.size
    present = ink_ratio > ratio_pixel
    return present, ink_ratio

