from skimage import measure

def find_regions(img_binaire):
    labels = measure.label(img_binaire, connectivity=2)
    item_regions = measure.regionprops(labels)
    return item_regions



def find_signature_box(item_regions, image_shape):
    H, W = image_shape
    signature_box_possibilities = []

    for p in item_regions:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc

        is_trop_large = w > 0.20 * W and h > 0.08 * H
        is_pas_assez_large = w < 0.50 * W and h < 0.25 * H
        ratio_correct = 1.2 < (w / h) < 3.5

        if is_trop_large and is_pas_assez_large and ratio_correct:
            signature_box_possibilities.append((minc, minr, maxc, maxr))

    if len(signature_box_possibilities) == 0:
        return None

    x1, y1, x2, y2 = signature_box_possibilities[-1]

    return x1, y1, x2, y2



def find_square_boxes(item_regions):
    boxes = []

    for p in item_regions:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        if 20 <= w <= 45 and 20 <= h <= 45:
            ratio = w / h

            if 0.7 <= ratio <= 1.3 and area >= 180:
                boxes.append({"bbox": (minc, minr, maxc, maxr),"cx": (minc + maxc) / 2,"cy": (minr + maxr) / 2,"area": area,"w": w,"h": h})

    return boxes


