def detect_if_the_signature_is_there(img_binaire, signature_box, ratio_pixel):
    
    if signature_box is None:
        return False, 0.0

    x1, y1, x2, y2 = signature_box

    w = x2 - x1
    h = y2 - y1

    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    inside = img_binaire[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

    ink_ratio = inside.sum() / inside.size
    present = ink_ratio > ratio_pixel

    return present, ink_ratio


def ink_ratio_inside_box(img_binaire, bbox, pad_ratio=0.22):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = int(pad_ratio * w)
    pad_y = int(pad_ratio * h)

    inner = img_binaire[
        y1 + pad_y:y2 - pad_y,
        x1 + pad_x:x2 - pad_x
    ]

    if inner.size == 0:
        return 0.0

    return inner.sum() / inner.size