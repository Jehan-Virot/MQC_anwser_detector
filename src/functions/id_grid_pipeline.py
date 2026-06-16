from skimage import morphology, measure
import itertools
import numpy as np
from functions.outils_generaux_images import load_gray_image, binarize_dark_pixels_for_id

ID_ROI_RATIO = (0.66, 0.10, 0.99, 0.56) 

def crop_id_roi(img, roi_ratio=ID_ROI_RATIO):
    H, W = img.shape[:2]
    rx1, ry1, rx2, ry2 = roi_ratio
    x1 = int(rx1 * W)
    y1 = int(ry1 * H)
    x2 = int(rx2 * W)
    y2 = int(ry2 * H)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def find_square_candidates(img_bin):
    """Trouve des candidats de cases carrées dans la ROI Student ID."""
    roi, _ = crop_id_roi(img_bin)
    roi_clean = morphology.closing(roi, morphology.footprint_rectangle((2,2)))

    labels = measure.label(roi_clean, connectivity=2)
    props = measure.regionprops(labels)

    H, W = roi.shape
    candidates = []

    for p in props:
        minr, minc, maxr, maxc = p.bbox
        h = maxr - minr
        w = maxc - minc
        area = p.area

        min_side = max(10, int(0.015 * H))
        max_side = max(25, int(0.110 * H))

        ratio = w / h if h > 0 else 0
        extent = area / float(w * h) if w * h > 0 else 0

        if min_side <= w <= max_side and min_side <= h <= max_side:
            if 0.65 <= ratio <= 1.45 and 0.08 <= extent <= 0.75:
                candidates.append({
                    "bbox": (minc, minr, maxc, maxr),
                    "cx": (minc + maxc) / 2.0,
                    "cy": (minr + maxr) / 2.0,
                    "w": w,
                    "h": h,
                    "area": area,
                })

    return candidates


def group_values(values, max_gap):
    if len(values) == 0:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - np.mean(groups[-1])) <= max_gap:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [{"center": float(np.mean(g)), "count": len(g), "values": g} for g in groups]


def select_regular_clusters(clusters, n_expected):
    """Sélectionne n_expected groupes formant une grille régulière."""
    if len(clusters) < n_expected:
        return None

    clusters = sorted(clusters, key=lambda c: c["center"])
    best_combo = None
    best_cost = float("inf")
    max_count = max(c["count"] for c in clusters)

    for combo in itertools.combinations(clusters, n_expected):
        centers = np.array([c["center"] for c in combo], dtype=float)
        gaps = np.diff(centers)
        if np.any(gaps <= 0):
            continue

        mean_gap = np.mean(gaps)
        if mean_gap <= 0:
            continue

        regularity_cost = np.std(gaps) / mean_gap
        count_bonus = sum(c["count"] for c in combo) / float(n_expected * max_count)
        cost = regularity_cost - 0.10 * count_bonus

        if cost < best_cost:
            best_cost = cost
            best_combo = combo

    if best_combo is None:
        return None

    return sorted([float(c["center"]) for c in best_combo])


def fit_affine_grid_from_candidates(candidates, image_shape):
    """
    Reconstruit la grille comme un petit repère affine.

    Avant, on prenait uniquement des lignes/colonnes parfaitement horizontales et verticales.
    Si la photo est prise de travers, les centres des cases forment plutôt un parallélogramme.
    Ici on estime donc :
        centre(row, col) = origine + col * vecteur_colonne + row * vecteur_ligne
    Cela corrige les décalages progressifs qui faisaient tomber l'intérieur d'une case entre 2 cases.
    """
    if len(candidates) < 12:
        return None

    H_img, W_img = image_shape
    _, (rx1, ry1, _, _) = crop_id_roi(np.zeros(image_shape, dtype=bool))

    sides = np.array([0.5 * (c["w"] + c["h"]) for c in candidates], dtype=float)
    median_side = float(np.median(sides))
    group_tol = max(6, 0.75 * median_side)

    x_groups = group_values([c["cx"] for c in candidates], group_tol)
    y_groups = group_values([c["cy"] for c in candidates], group_tol)

    x_centers = select_regular_clusters(x_groups, 5)
    y_centers = select_regular_clusters(y_groups, 10)
    if x_centers is None or y_centers is None:
        return None

    assigned = []
    for c in candidates:
        col = int(np.argmin([abs(c["cx"] - xc) for xc in x_centers]))
        row = int(np.argmin([abs(c["cy"] - yc) for yc in y_centers]))
        if abs(c["cx"] - x_centers[col]) <= 1.4 * group_tol and abs(c["cy"] - y_centers[row]) <= 1.4 * group_tol:
            assigned.append((row, col, c["cx"], c["cy"]))

    if len(assigned) < 12:
        return None

    # Matrice de moindres carrés : x = a0 + a1*col + a2*row, y = b0 + b1*col + b2*row
    A = np.array([[1.0, col, row] for row, col, _, _ in assigned], dtype=float)
    bx = np.array([x for _, _, x, _ in assigned], dtype=float)
    by = np.array([y for _, _, _, y in assigned], dtype=float)

    coef_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Qualité du modèle : si l'erreur est énorme, on ne force pas.
    pred_x = A @ coef_x
    pred_y = A @ coef_y
    err = np.sqrt((pred_x - bx) ** 2 + (pred_y - by) ** 2)
    if float(np.median(err)) > 0.75 * median_side:
        return None

    side = median_side * 1.08
    grid = []
    for row in range(10):
        line = []
        for col in range(5):
            x = coef_x[0] + coef_x[1] * col + coef_x[2] * row
            y = coef_y[0] + coef_y[1] * col + coef_y[2] * row
            x1 = int(round(rx1 + x - side / 2))
            y1 = int(round(ry1 + y - side / 2))
            x2 = int(round(rx1 + x + side / 2))
            y2 = int(round(ry1 + y + side / 2))
            x1 = max(0, min(W_img - 1, x1))
            x2 = max(0, min(W_img, x2))
            y1 = max(0, min(H_img - 1, y1))
            y2 = max(0, min(H_img, y2))
            line.append((x1, y1, x2, y2))
        grid.append(line)

    return grid


def reconstruct_id_grid(img_bin):
    """
    Retourne une grille 10 x 5 de bounding boxes absolues.
    Version robuste : elle utilise un modèle affine de grille, donc elle tolère une petite rotation
    ou perspective de la photo.
    """
    candidates = find_square_candidates(img_bin)
    grid = fit_affine_grid_from_candidates(candidates, img_bin.shape)
    return grid, candidates


def ink_ratio_inside_box(img_bin, bbox, pad_ratio=0.27):
    H, W = img_bin.shape
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))

    w = x2 - x1
    h = y2 - y1
    if w <= 4 or h <= 4:
        return 0.0

    px = int(pad_ratio * w)
    py = int(pad_ratio * h)

    inner = img_bin[y1 + py:y2 - py, x1 + px:x2 - px]
    if inner.size == 0:
        return 0.0

    return float(np.sum(inner)) / float(inner.size)


def read_student_id_from_grid(img_bin, grid):
    ratios = np.zeros((10, 5), dtype=float)

    for row in range(10):
        for col in range(5):
            ratios[row, col] = ink_ratio_inside_box(img_bin, grid[row][col])

    digits = []
    confidence = []

    for col in range(5):
        col_ratios = ratios[:, col]
        best_row = int(np.argmax(col_ratios))
        sorted_ratios = np.sort(col_ratios)
        best = float(sorted_ratios[-1])
        second = float(sorted_ratios[-2]) if len(sorted_ratios) >= 2 else 0.0

        dynamic_threshold = max(
            0.03,
            float(np.mean(col_ratios) + 1.6 * np.std(col_ratios))
        )

        if best >= dynamic_threshold and best >= second + 0.010:
            digits.append(str(best_row))
            confidence.append(best - second)
        else:
            digits.append("?")
            confidence.append(best - second)

    return "".join(digits), ratios, confidence


def read_student_id_grid_from_path(path):
    img_gray = load_gray_image(path)

    best_result = None
    # Plusieurs seuils : robuste aux photos très claires/sombres.
    # On garde le résultat qui donne une grille + le moins de cases ambiguës + la meilleure confiance.
    for threshold_factor in (0.82, 0.88, 0.94, 1.00, 1.06):
        img_bin = binarize_dark_pixels_for_id(img_gray, threshold_factor=threshold_factor)
        grid, candidates = reconstruct_id_grid(img_bin)
        if grid is None:
            current = {
                "student_id": None,
                "raw_student_id": None,
                "status": "grille_non_detectee",
                "grid": None,
                "ratios": None,
                "confidence": [],
                "img_gray": img_gray,
                "img_bin": img_bin,
                "candidates": candidates,
                "score": -9999,
            }
        else:
            student_id, ratios, confidence = read_student_id_from_grid(img_bin, grid)
            n_unknown = student_id.count("?")
            score = 100 - 20 * n_unknown + 100 * float(np.mean(confidence))
            current = {
                "student_id": None if n_unknown else student_id,
                "raw_student_id": student_id,
                "status": "ok" if n_unknown == 0 else "case_ambigue_" + student_id,
                "grid": grid,
                "ratios": ratios,
                "confidence": confidence,
                "img_gray": img_gray,
                "img_bin": img_bin,
                "candidates": candidates,
                "score": score,
                "threshold_factor": threshold_factor,
            }

        if best_result is None or current["score"] > best_result["score"]:
            best_result = current

    if best_result is not None:
        best_result.pop("score", None)
    return best_result

