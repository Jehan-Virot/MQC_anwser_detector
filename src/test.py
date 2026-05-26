from skimage import io, color, transform, exposure, filters, measure
import matplotlib.pyplot as plt
import numpy as np

from os import listdir, mkdir
from os.path import exists
from shutil import rmtree
import re


# ============================================================
# PARAMÈTRES
# ============================================================

SOURCE_PATH_DATA = "data/"
SIGNATURE_MIN_INK_RATIO = 0.006

DEBUG_ID = True
GAUSSIAN_SIGMA = 1.0


# ============================================================
# CHARGEMENT / BINARISATION
# ============================================================

def load_binary_image(path):
    try:
        print(f"image : {path}")
        img = io.imread(path)

        img_gray = rgb_to_grey_and_or_normalisation(img)
        img_gray = turn_image_if_needed(img_gray)

        # Flou gaussien avant stretching + Otsu
        img_gray = filters.gaussian(
            img_gray,
            sigma=GAUSSIAN_SIGMA,
            preserve_range=True
        )

        img_gray = stretch_histo(img_gray)

        threshold = filters.threshold_otsu(img_gray)

        # Pixels noirs = True
        img_binaire = img_gray < threshold

        return img_binaire.astype(bool)

    except Exception as e:
        print("Erreur load_binary_image:", e)
        return None


def rgb_to_grey_and_or_normalisation(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)
    else:
        img = img.astype(float)
        if img.max() > 1.0:
            img = img / 255.0

    return img


def turn_image_if_needed(img):
    H, W = img.shape

    if W > H:
        img = transform.rotate(
            img,
            angle=270,
            resize=True,
            preserve_range=True
        )

    return img


def stretch_histo(img):
    low, high = np.percentile(img, (5, 95))

    if high <= low:
        return img

    return exposure.rescale_intensity(
        img,
        in_range=(low, high),
        out_range=(0, 1)
    )


# ============================================================
# MORPHOLOGIE MANUELLE
# ============================================================

def binary_erosion_manual(img, iterations=1):
    """
    Erosion binaire 3x3 manuelle.
    Ici True = encre noire.
    On l'utilise seulement pour trouver les labels, pas pour lire l'encre.
    """
    out = img.copy()

    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)

        result = np.ones_like(out, dtype=bool)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                result = result & padded[
                    1 + dy:1 + dy + out.shape[0],
                    1 + dx:1 + dx + out.shape[1]
                ]

        out = result

    return out


# ============================================================
# LABELING / SIGNATURE
# ============================================================

def find_regions(img_binaire):
    labels = measure.label(img_binaire, connectivity=2)
    return measure.regionprops(labels)


def find_signature_box(item_regions, image_shape):
    H, W = image_shape
    signature_box_possibilities = []

    for p in item_regions:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc

        if h == 0:
            continue

        is_trop_large = w > 0.20 * W and h > 0.08 * H
        is_pas_assez_large = w < 0.55 * W and h < 0.30 * H
        ratio_correct = 1.2 < (w / h) < 4.0

        if is_trop_large and is_pas_assez_large and ratio_correct:
            signature_box_possibilities.append((minc, minr, maxc, maxr))

    if len(signature_box_possibilities) == 0:
        return None

    signature_box_possibilities = sorted(
        signature_box_possibilities,
        key=lambda b: b[1]
    )

    return signature_box_possibilities[-1]


def detect_if_the_signature_is_there(img_binaire, signature_box, ratio_pixel):
    if signature_box is None:
        return False, 0.0

    x1, y1, x2, y2 = signature_box

    w = x2 - x1
    h = y2 - y1

    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    inside = img_binaire[
        y1 + pad_y:y2 - pad_y,
        x1 + pad_x:x2 - pad_x
    ]

    if inside.size == 0:
        return False, 0.0

    ink_ratio = inside.sum() / inside.size
    present = ink_ratio > ratio_pixel

    return present, ink_ratio


# ============================================================
# CANDIDATS STUDENT ID
# ============================================================

def find_square_boxes(img_binaire, image_shape, debug=False):
    H, W = image_shape

    # ROI Student ID : volontairement plus stricte pour éviter GROUP et EXAM CONDITIONS
    x_min = int(0.66 * W)
    x_max = int(0.96 * W)
    y_min = int(0.13 * H)
    y_max = int(0.55 * H)

    roi = img_binaire[y_min:y_max, x_min:x_max]

    # Image seulement pour le labeling
    roi_for_labels = binary_erosion_manual(roi, iterations=1)

    labels = measure.label(roi_for_labels, connectivity=2)
    regions = measure.regionprops(labels)

    boxes = []
    
    for p in regions:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        if h == 0 or w == 0:
            continue

        ratio = w / h
        extent = area / (w * h)

        # Après érosion, les labels sont plus petits.
        good_size = 5 <= w <= 55 and 5 <= h <= 55
        good_ratio = 0.35 <= ratio <= 2.80
        good_area = 5 <= area <= 1000
        good_extent = 0.02 <= extent <= 0.98

        if good_size and good_ratio and good_area and good_extent:
            boxes.append({
                "bbox": (
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
                "extent": extent
            })

    return boxes


# ============================================================
# LECTURE D'ENCRE
# ============================================================

def ink_ratio_inside_box(img_binaire, bbox, pad_ratio=0.25):
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


def ink_ratio_around_center(img_binaire, cx, cy, box_w, box_h, pad_ratio=0.30):
    x1 = int(cx - box_w / 2)
    x2 = int(cx + box_w / 2)
    y1 = int(cy - box_h / 2)
    y2 = int(cy + box_h / 2)

    H, W = img_binaire.shape

    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)

    return ink_ratio_inside_box(
        img_binaire,
        (x1, y1, x2, y2),
        pad_ratio=pad_ratio
    )


# ============================================================
# SCORE TEMPLATE CASE
# ============================================================

def square_template_score(img_binaire, cx, cy, box_w, box_h):
    """
    Score élevé si la fenêtre autour du centre ressemble à une case :
    présence d'encre sur les quatre bords.
    """
    H, W = img_binaire.shape

    x1 = int(cx - box_w / 2)
    x2 = int(cx + box_w / 2)
    y1 = int(cy - box_h / 2)
    y2 = int(cy + box_h / 2)

    if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
        return 0.0

    patch = img_binaire[y1:y2, x1:x2]

    if patch.size == 0:
        return 0.0

    ph, pw = patch.shape

    if ph < 8 or pw < 8:
        return 0.0

    t = max(1, int(0.15 * min(ph, pw)))

    top = patch[:t, :]
    bottom = patch[-t:, :]
    left = patch[:, :t]
    right = patch[:, -t:]

    center = patch[
        int(0.30 * ph):int(0.70 * ph),
        int(0.30 * pw):int(0.70 * pw)
    ]

    top_score = top.mean()
    bottom_score = bottom.mean()
    left_score = left.mean()
    right_score = right.mean()
    center_score = center.mean() if center.size > 0 else 0.0

    border_score = (
        top_score +
        bottom_score +
        left_score +
        right_score
    ) / 4

    side_balance = min(top_score, bottom_score, left_score, right_score)

    score = 0.75 * border_score + 0.25 * side_balance

    if side_balance > 0.08:
        score += 0.15

    # Malus si on est plutôt sur un chiffre/texte plein que sur une case
    if center_score > border_score * 2.5 and center_score > 0.20:
        score *= 0.5

    return float(score)


# ============================================================
# RECONSTRUCTION GRILLE 5 x 10
# ============================================================

def estimate_box_size_from_candidates(candidates):
    widths = [b["w"] for b in candidates if 5 <= b["w"] <= 65]
    heights = [b["h"] for b in candidates if 5 <= b["h"] <= 65]

    if len(widths) == 0:
        box_w = 28
    else:
        box_w = float(np.median(widths)) * 1.35

    if len(heights) == 0:
        box_h = 28
    else:
        box_h = float(np.median(heights)) * 1.35

    box_w = max(18, min(45, box_w))
    box_h = max(18, min(45, box_h))

    return box_w, box_h


def estimate_grid_steps(candidates, debug=False):
    horizontal_distances = []
    vertical_distances = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            b1 = candidates[i]
            b2 = candidates[j]

            dx = abs(b1["cx"] - b2["cx"])
            dy = abs(b1["cy"] - b2["cy"])

            avg_h = (b1["h"] + b2["h"]) / 2
            avg_w = (b1["w"] + b2["w"]) / 2

            # voisins horizontaux probables
            if dy < 0.75 * avg_h and 12 <= dx <= 80:
                horizontal_distances.append(dx)

            # voisins verticaux probables
            if dx < 0.75 * avg_w and 12 <= dy <= 85:
                vertical_distances.append(dy)

    def robust_step(values, default_value):
        if len(values) == 0:
            return default_value

        values = np.array(values)

        rounded = np.round(values / 2) * 2
        unique, counts = np.unique(rounded, return_counts=True)

        best = unique[np.argmax(counts)]
        close_values = values[np.abs(values - best) <= 6]

        if len(close_values) == 0:
            return float(best)

        return float(np.median(close_values))

    dx_step = robust_step(horizontal_distances, default_value=33)
    dy_step = robust_step(vertical_distances, default_value=38)

    return dx_step, dy_step


def find_nearest_candidate(candidates, x, y, max_dist):
    best = None
    best_dist = float("inf")

    for b in candidates:
        d = ((b["cx"] - x) ** 2 + (b["cy"] - y) ** 2) ** 0.5

        if d < best_dist:
            best_dist = d
            best = b

    if best is not None and best_dist <= max_dist:
        return best, best_dist

    return None, best_dist


def score_grid(img_binaire, candidates, origin_x, origin_y, dx_step, dy_step, box_w, box_h):
    matched = []
    matched_ids = set()

    label_score = 0.0
    template_score = 0.0

    tolerance = max(10, 0.38 * min(dx_step, dy_step))

    for row in range(10):
        for col in range(5):
            x = origin_x + col * dx_step
            y = origin_y + row * dy_step

            s = square_template_score(img_binaire, x, y, box_w, box_h)
            template_score += s

            b, dist = find_nearest_candidate(candidates, x, y, tolerance)

            if b is not None:
                b_id = id(b)

                if b_id not in matched_ids:
                    matched_ids.add(b_id)
                    matched.append((row, col, b))

                    label_score += 1.0 - 0.35 * (dist / tolerance)

    final_score = label_score + 2.2 * template_score

    return final_score, matched


def validate_grid_neighbors(matched, min_neighbors=2):
    occupied = set((row, col) for row, col, _ in matched)

    valid_points = 0

    for row, col, _ in matched:
        neighbors = 0

        if (row - 1, col) in occupied:
            neighbors += 1
        if (row + 1, col) in occupied:
            neighbors += 1
        if (row, col - 1) in occupied:
            neighbors += 1
        if (row, col + 1) in occupied:
            neighbors += 1

        if neighbors >= min_neighbors:
            valid_points += 1

    return valid_points


def refine_grid_origin(img_binaire, candidates, origin_x, origin_y, dx_step, dy_step, box_w, box_h):
    best_score = -1
    best_origin = (origin_x, origin_y)
    best_matched = []

    for ox_shift in range(-8, 9, 2):
        for oy_shift in range(-8, 9, 2):
            test_origin_x = origin_x + ox_shift
            test_origin_y = origin_y + oy_shift

            score, matched = score_grid(
                img_binaire,
                candidates,
                test_origin_x,
                test_origin_y,
                dx_step,
                dy_step,
                box_w,
                box_h
            )

            neighbor_score = validate_grid_neighbors(matched, min_neighbors=2)
            final_score = score + 0.35 * neighbor_score

            if final_score > best_score:
                best_score = final_score
                best_origin = (test_origin_x, test_origin_y)
                best_matched = matched

    return best_origin[0], best_origin[1], best_score, best_matched


def find_student_id_grid(img_binaire, square_boxes, image_shape, debug=False):
    H, W = image_shape

    candidates = [
        b for b in square_boxes
        if b["cx"] > 0.64 * W
        and 0.12 * H < b["cy"] < 0.56 * H
    ]


    if len(candidates) < 10:
        return None, None, None, None

    dx_step, dy_step = estimate_grid_steps(candidates, debug=debug)
    box_w, box_h = estimate_box_size_from_candidates(candidates)

    best_score = -1
    best_grid = None
    best_matched = None

    # Chaque candidat peut être n'importe quelle case de la grille 10x5.
    for b in candidates:
        for row_guess in range(10):
            for col_guess in range(5):
                origin_x = b["cx"] - col_guess * dx_step
                origin_y = b["cy"] - row_guess * dy_step

                score, matched = score_grid(
                    img_binaire,
                    candidates,
                    origin_x,
                    origin_y,
                    dx_step,
                    dy_step,
                    box_w,
                    box_h
                )

                neighbor_score = validate_grid_neighbors(
                    matched,
                    min_neighbors=2
                )

                final_score = score + 0.35 * neighbor_score

                if final_score > best_score:
                    best_score = final_score
                    best_grid = (origin_x, origin_y, dx_step, dy_step)
                    best_matched = matched

    if best_grid is None:
        return None, None, None, None

    origin_x, origin_y, dx_step, dy_step = best_grid

    origin_x, origin_y, best_score, best_matched = refine_grid_origin(
        img_binaire,
        candidates,
        origin_x,
        origin_y,
        dx_step,
        dy_step,
        box_w,
        box_h
    )

    x_centers = [origin_x + col * dx_step for col in range(5)]
    y_centers = [origin_y + row * dy_step for row in range(10)]


    if len(best_matched) < 18:
        return None, None, None, None

    grid_model = {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "dx_step": dx_step,
        "dy_step": dy_step,
        "matched": best_matched,
        "box_w": box_w,
        "box_h": box_h
    }

    return candidates, x_centers, y_centers, grid_model


def estimate_box_size(candidates):
    return estimate_box_size_from_candidates(candidates)


def read_student_id(img_binaire, square_boxes, image_shape, min_ratio=1.45, debug=False):
    candidates, x_centers, y_centers, grid_model = find_student_id_grid(
        img_binaire,
        square_boxes,
        image_shape,
        debug=debug
    )

    if candidates is None:
        return None, None

    box_w = grid_model["box_w"]
    box_h = grid_model["box_h"]

    matrix = np.zeros((10, 5))

    for row, y in enumerate(y_centers):
        for col, x in enumerate(x_centers):
            matrix[row, col] = ink_ratio_around_center(
                img_binaire,
                x,
                y,
                box_w,
                box_h,
                pad_ratio=0.30
            )

    digits = []

    for col in range(5):
        column_values = matrix[:, col]

        best_row = int(np.argmax(column_values))
        best_value = column_values[best_row]
        median_value = float(np.median(column_values))
        sorted_values = sorted(column_values, reverse=True)
        second_best = sorted_values[1]

        condition_absolute = best_value > 0.025
        condition_median = best_value > max(0.015, median_value * min_ratio)
        condition_second = best_value > second_best * 1.08 or best_value > 0.08

        if condition_absolute and condition_median and condition_second:
            digits.append(str(best_row))
        else:
            digits.append("?")

    student_id = "".join(digits)

    grid_info = {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "matrix": matrix,
        "box_w": box_w,
        "box_h": box_h,
        "grid_model": grid_model
    }


    return student_id, grid_info


# ============================================================
# DESSIN DEBUG
# ============================================================

def draw_results(img, signature_box, grid_info, output_path="result_detection.png"):
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img, cmap="gray")

    if signature_box is not None:
        x1, y1, x2, y2 = signature_box

        ax.plot(
            [x1, x2, x2, x1, x1],
            [y1, y1, y2, y2, y1],
            linewidth=3
        )

        ax.text(x1, y1 - 15, "Bloc signature", fontsize=12)

    if grid_info is not None:
        x_centers = grid_info["x_centers"]
        y_centers = grid_info["y_centers"]

        box_w = grid_info.get("box_w", 25)
        box_h = grid_info.get("box_h", 25)

        for row, y in enumerate(y_centers):
            for col, x in enumerate(x_centers):
                ax.plot(x, y, marker="o", markersize=4)

                x1 = x - box_w / 2
                x2 = x + box_w / 2
                y1 = y - box_h / 2
                y2 = y + box_h / 2

                ax.plot(
                    [x1, x2, x2, x1, x1],
                    [y1, y1, y2, y2, y1],
                    linewidth=1
                )

            ax.text(x_centers[0] - 40, y, str(row), fontsize=10)

        ax.text(
            x_centers[0],
            y_centers[0] - 30,
            "Grille Student ID",
            fontsize=12
        )

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print("Image sauvegardée :", output_path)


# ============================================================
# MAIN
# ============================================================

def reset_output_folder(path):
    if exists(path):
        rmtree(path)

    mkdir(path)


def main():
    reset_output_folder("signature_fails")
    reset_output_folder("id_fails")

    source_dir = listdir(SOURCE_PATH_DATA)
    source_dir = [form for form in source_dir if re.match(r"^FORM\d$", form)]

    list_etudiant_valide = []
    list_etudiant_non_valide = []
    list_etudiant_signature = []
    list_etudiant_non_signature = []

    for form in source_dir:
        temp_dir_path = f"{SOURCE_PATH_DATA}{form}/EXAM_{form}_PRESENCES/"

        if not exists(temp_dir_path):
            continue

        presence_pages = listdir(temp_dir_path)

        for current_presence_page in presence_pages:
            current_presence_page_path = f"{temp_dir_path}{current_presence_page}"

            img_binaire = load_binary_image(current_presence_page_path)

            if img_binaire is None:
                continue

            items_regions = find_regions(img_binaire)

            signature_box = find_signature_box(
                items_regions,
                img_binaire.shape
            )

            has_signature, ink_ratio = detect_if_the_signature_is_there(
                img_binaire,
                signature_box,
                SIGNATURE_MIN_INK_RATIO
            )

            square_boxes = find_square_boxes(
                img_binaire,
                img_binaire.shape,
                debug=DEBUG_ID
            )

            student_id, grid_info = read_student_id(
                img_binaire,
                square_boxes,
                img_binaire.shape,
                debug=DEBUG_ID
            )


            if student_id is None or "?" in student_id:
                list_etudiant_non_valide.append((student_id, current_presence_page))

                draw_results(
                    img_binaire,
                    signature_box,
                    grid_info,
                    f"id_fails/{current_presence_page}"
                )

            elif current_presence_page.find(student_id) != -1:
                list_etudiant_valide.append(student_id)

            else:
                list_etudiant_non_valide.append((student_id, current_presence_page))

                draw_results(
                    img_binaire,
                    signature_box,
                    grid_info,
                    f"id_fails/{current_presence_page}"
                )

            if has_signature:
                list_etudiant_signature.append(student_id)

            else:
                list_etudiant_non_signature.append((student_id, current_presence_page))

                draw_results(
                    img_binaire,
                    signature_box,
                    grid_info,
                    f"signature_fails/{current_presence_page}"
                )

    total_id = len(list_etudiant_valide) + len(list_etudiant_non_valide)
    total_signature = len(list_etudiant_signature) + len(list_etudiant_non_signature)

    print("\n========== RÉSULTATS ==========")

    print(f"nb étudiant avec ID détecté : {len(list_etudiant_valide)}/{total_id}")
    print(f"nb étudiant avec ID non détecté : {len(list_etudiant_non_valide)}/{total_id}")

    print(
        f"nb étudiant avec signature détectée : "
        f"{len(list_etudiant_signature)}/{total_signature}"
    )
    print(
        f"nb étudiant avec signature non détectée : "
        f"{len(list_etudiant_non_signature)}/{total_signature}"
    )

    print("liste étudiant id detect :", list_etudiant_valide)
    print("liste étudiant non id detect :", list_etudiant_non_valide)
    print("liste étudiant signature detect :", list_etudiant_signature)
    print("liste étudiant non signature detect :", list_etudiant_non_signature)


if __name__ == "__main__":
    main()