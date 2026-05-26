from skimage import measure, morphology
from functions.item_region_funcs import remove_small_objects

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



def find_square_boxes(img_binaire, image_shape, debug=True):
    H, W = image_shape

    # Zone Student ID approximative
    x_min = int(0.62 * W)
    x_max = int(0.90 * W)
    y_min = int(0.15 * H)
    y_max = int(0.48 * H)

    roi_original = img_binaire[y_min:y_max, x_min:x_max]

    # Debug AVANT nettoyage
    labels_before = measure.label(roi_original, connectivity=2)
    regions_before = measure.regionprops(labels_before)

    if debug:
        print("\n========== DEBUG find_square_boxes ==========")
        print("Image shape:", image_shape)
        print("ROI coords:", {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max
        })
        print("ROI shape:", roi_original.shape)
        print("Régions AVANT nettoyage:", len(regions_before))

    # Nettoyage léger
    # Attention : min_size=40 peut supprimer des morceaux de cases si les contours sont cassés.
    roi = remove_small_objects(roi_original, min_size=40)
    roi = morphology.closing(roi, morphology.footprint_rectangle((2, 2)))

    labels = measure.label(roi, connectivity=2)
    regions = measure.regionprops(labels)

    if debug:
        print("Régions APRÈS nettoyage:", len(regions))

    boxes = []
    all_infos = []

    # Compteurs pour comprendre où ça bloque
    count_total = 0
    count_size_ok = 0
    count_ratio_ok = 0
    count_extent_ok = 0
    count_all_ok = 0

    reject_reasons = {
        "too_small_or_big": 0,
        "bad_ratio": 0,
        "bad_extent": 0
    }

    for p in regions:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        if h == 0 or w == 0:
            continue

        count_total += 1

        ratio = w / h
        bbox_area = w * h
        extent = area / bbox_area

        good_size = 14 <= w <= 35 and 14 <= h <= 35
        good_ratio = 0.75 <= ratio <= 1.25
        good_extent = 0.15 <= extent <= 0.85

        if good_size:
            count_size_ok += 1
        if good_ratio:
            count_ratio_ok += 1
        if good_extent:
            count_extent_ok += 1

        if not good_size:
            reject_reasons["too_small_or_big"] += 1
        elif not good_ratio:
            reject_reasons["bad_ratio"] += 1
        elif not good_extent:
            reject_reasons["bad_extent"] += 1

        info = {
            "bbox_roi": (minc, minr, maxc, maxr),
            "bbox_img": (
                minc + x_min,
                minr + y_min,
                maxc + x_min,
                maxr + y_min
            ),
            "cx": (minc + maxc) / 2 + x_min,
            "cy": (minr + maxr) / 2 + y_min,
            "area": area,
            "w": w,
            "h": h,
            "ratio": ratio,
            "extent": extent,
            "good_size": good_size,
            "good_ratio": good_ratio,
            "good_extent": good_extent
        }

        all_infos.append(info)

        if good_size and good_ratio and good_extent:
            count_all_ok += 1

            boxes.append({
                "bbox": info["bbox_img"],
                "cx": info["cx"],
                "cy": info["cy"],
                "area": area,
                "w": w,
                "h": h,
                "ratio": ratio,
                "extent": extent
            })

    if debug:
        print("\n--- Résumé filtres ---")
        print("Total régions analysées:", count_total)
        print("Passent good_size:", count_size_ok)
        print("Passent good_ratio:", count_ratio_ok)
        print("Passent good_extent:", count_extent_ok)
        print("Passent TOUS les critères:", count_all_ok)
        print("Raisons principales de rejet:", reject_reasons)

        print("\n--- Top 40 régions APRÈS nettoyage par area ---")
        all_infos_sorted = sorted(all_infos, key=lambda r: r["area"], reverse=True)

        for i, r in enumerate(all_infos_sorted[:40]):
            print(
                f"{i:02d} | "
                f"w={r['w']:3d}, h={r['h']:3d}, area={r['area']:5.0f}, "
                f"ratio={r['ratio']:.2f}, extent={r['extent']:.2f}, "
                f"size={r['good_size']}, ratio_ok={r['good_ratio']}, extent_ok={r['good_extent']}, "
                f"bbox_roi={r['bbox_roi']}"
            )

        print("\n--- Boxes gardées ---")
        for i, b in enumerate(boxes):
            print(
                f"{i:02d} | "
                f"cx={b['cx']:.1f}, cy={b['cy']:.1f}, "
                f"w={b['w']}, h={b['h']}, area={b['area']}, "
                f"ratio={b['ratio']:.2f}, extent={b['extent']:.2f}, "
                f"bbox={b['bbox']}"
            )

        print("Nombre final de boxes:", len(boxes))
        print("==============================================\n")

    return boxes