# main_merged_student_id_signature_cnn.py
# Programme complet :
# 1) lecture Student ID via grille 5 x 10
# 2) détection signature_box avec l'ancienne méthode qui fonctionnait mieux
# 3) sanitisation de la signature
# 4) reconnaissance signature avec model.keras + labels.json
# 5) génération presence_results.csv

from os import listdir, mkdir
from os.path import exists, isdir, join
from shutil import rmtree
import re
import csv
import json
import itertools
import numpy as np
import skimage as ski
import keras

from skimage import io, color, transform, exposure, filters, measure, morphology, draw


# ============================================================
# PARAMÈTRES
# ============================================================

SOURCE_PATH_DATA = "data/"

MODEL_PATH = "signature_model/model.keras"
LABELS_PATH = "signature_model/labels.json"

SIGNATURE_FAILS_DIR = "signature_fails"
ID_FAILS_DIR = "id_fails"
DEBUG_OK_DIR = "id_debug_ok"
OUTPUT_CSV = "presence_results.csv"

SAVE_OK_DEBUG = False

CNN_MIN_CONFIDENCE = 0.55

SIGNATURE_HEIGHT = 239
SIGNATURE_WIDTH = 342

ID_ROI_RATIO = (0.68, 0.12, 0.98, 0.55)

N_COLS = 5
N_ROWS = 10

MIN_CHECK_RATIO = 0.030
INNER_PAD_RATIO = 0.27

SIGNATURE_MIN_INK_RATIO = 0.006


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def ensure_clean_dir(path):
    if exists(path):
        rmtree(path)
    mkdir(path)


def rgb_to_gray_and_normalize(img):
    if img.ndim == 3:
        return color.rgb2gray(img)

    img = img.astype(float)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def turn_image_if_needed(img):
    h, w = img.shape[:2]

    if w > h:
        img = transform.rotate(
            img,
            angle=270,
            resize=True,
            preserve_range=True
        )

    return img


def stretch_histo(img_gray, p_low=5, p_high=95):
    low, high = np.percentile(img_gray, (p_low, p_high))

    if high <= low:
        return img_gray

    return exposure.rescale_intensity(
        img_gray,
        in_range=(low, high),
        out_range=(0, 1)
    )


def load_gray_image(path):
    img = io.imread(path)
    img_gray = rgb_to_gray_and_normalize(img)
    img_gray = turn_image_if_needed(img_gray)
    return img_gray


def load_binary_image(path):
    """
    Image entière :
    gris -> stretching -> Otsu -> binaire.
    True = pixels noirs / encre.
    """
    img_gray = load_gray_image(path)
    img_stretched = stretch_histo(img_gray)
    threshold = filters.threshold_otsu(img_stretched)
    img_binaire = img_stretched < threshold
    return img_binaire


def binarize_dark_pixels_for_id(img_gray):
    """
    Binarisation un peu plus stricte pour la grille ID.
    """
    img_stretched = stretch_histo(img_gray)
    otsu = filters.threshold_otsu(img_stretched)

    threshold = otsu * 0.92

    img_bin = img_stretched < threshold
    img_bin = morphology.remove_small_objects(img_bin, min_size=8)
    img_bin = morphology.binary_closing(img_bin, morphology.square(2))

    return img_bin


# ============================================================
# CNN SIGNATURE
# ============================================================

def load_signature_model_and_labels():
    if not exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    if not exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels introuvables : {LABELS_PATH}")

    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_to_id = json.load(f)

    label_index_to_student_id = {
        int(label): student_id
        for label, student_id in labels_to_id.items()
    }

    return model, label_index_to_student_id


def preprocess_signature_crop_for_cnn(signature_crop_gray):
    """
    Sanitisation de la signature extraite :
    - gris float [0,1]
    - stretching
    - filtre gaussien
    - Otsu
    - suppression légère du bruit
    - tentative suppression des composants collés au bord
    - resize 239 x 342
    - format CNN : (1, 239, 342, 1)
    """
    img = rgb_to_gray_and_normalize(signature_crop_gray)

    img = stretch_histo(img, p_low=3, p_high=97)

    img_smooth = filters.gaussian(img, sigma=1.0, preserve_range=True)

    threshold = filters.threshold_otsu(img_smooth)

    img_bin = img_smooth < threshold
    img_bin = morphology.remove_small_objects(img_bin, min_size=8)

    labels = measure.label(img_bin, connectivity=2)
    h, w = img_bin.shape

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[h - 1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, w - 1]))

    clean = img_bin.copy()

    for lab in border_labels:
        if lab != 0:
            clean[labels == lab] = False

    if clean.sum() < 10:
        clean = img_bin

    clean_float = clean.astype(np.float32)

    clean_resized = transform.resize(
        clean_float,
        (SIGNATURE_HEIGHT, SIGNATURE_WIDTH),
        preserve_range=True,
        anti_aliasing=False
    )

    clean_resized = clean_resized > 0.35
    clean_resized = clean_resized.astype(np.float32)

    x = clean_resized.reshape(1, SIGNATURE_HEIGHT, SIGNATURE_WIDTH, 1)

    return x, clean_resized


def predict_signature_student_id(model, label_index_to_student_id, signature_crop_gray):
    x, clean_binary = preprocess_signature_crop_for_cnn(signature_crop_gray)

    proba = model.predict(x, verbose=0)[0]

    best_label = int(np.argmax(proba))
    confidence = float(proba[best_label])

    if confidence < CNN_MIN_CONFIDENCE:
        return None, confidence, clean_binary

    student_id = label_index_to_student_id.get(best_label)

    return student_id, confidence, clean_binary


# ============================================================
# SIGNATURE : RETOUR EN ARRIÈRE SUR LA BOX
# ============================================================

def find_regions(img_binaire):
    labels = measure.label(img_binaire, connectivity=2)
    return measure.regionprops(labels)


def find_signature_box(item_regions, image_shape):
    """
    Ancienne méthode de détection de la signature_box.
    C'est volontairement la version qui fonctionnait avant.
    """
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

    x1, y1, x2, y2 = signature_box_possibilities[-1]

    return x1, y1, x2, y2


def crop_signature_inside_box(img_gray, signature_box):
    """
    Crop léger de l'intérieur de la boîte.
    La détection de la box reste l'ancienne.
    """
    if signature_box is None:
        return None

    x1, y1, x2, y2 = signature_box

    H, W = img_gray.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    crop = img_gray[
        y1 + pad_y:y2 - pad_y,
        x1 + pad_x:x2 - pad_x
    ]

    if crop.size == 0:
        return None

    return crop


def detect_if_the_signature_is_there(img_binaire, signature_box, ratio_pixel=SIGNATURE_MIN_INK_RATIO):
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
# DÉTECTION STUDENT ID : GRILLE 5 x 10
# ============================================================

def crop_id_roi(img):
    h, w = img.shape[:2]

    rx1, ry1, rx2, ry2 = ID_ROI_RATIO

    x1 = int(rx1 * w)
    y1 = int(ry1 * h)
    x2 = int(rx2 * w)
    y2 = int(ry2 * h)

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def find_square_candidates(img_bin):
    roi, _ = crop_id_roi(img_bin)

    roi_clean = morphology.binary_closing(roi, morphology.square(2))

    labels = measure.label(roi_clean, connectivity=2)
    props = measure.regionprops(labels)

    h_roi, w_roi = roi.shape

    candidates = []

    for p in props:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        min_side = max(10, int(0.015 * h_roi))
        max_side = max(25, int(0.110 * h_roi))

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
                    "area": area
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

    return [
        {
            "center": float(np.mean(g)),
            "count": len(g),
            "values": g
        }
        for g in groups
    ]


def select_regular_clusters(clusters, n_expected):
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


def reconstruct_id_grid(img_bin):
    candidates = find_square_candidates(img_bin)
    _, (rx1, ry1, _, _) = crop_id_roi(img_bin)

    if len(candidates) < 15:
        return None, candidates

    sides = [0.5 * (c["w"] + c["h"]) for c in candidates]
    median_side = float(np.median(sides))

    group_tol = max(6, 0.65 * median_side)

    x_groups = group_values([c["cx"] for c in candidates], group_tol)
    y_groups = group_values([c["cy"] for c in candidates], group_tol)

    x_centers = select_regular_clusters(x_groups, N_COLS)
    y_centers = select_regular_clusters(y_groups, N_ROWS)

    if x_centers is None or y_centers is None:
        return None, candidates

    side = median_side * 1.05

    grid = []

    for row in range(N_ROWS):
        line = []

        for col in range(N_COLS):
            cx = x_centers[col] + rx1
            cy = y_centers[row] + ry1

            x1 = int(cx - side / 2)
            y1 = int(cy - side / 2)
            x2 = int(cx + side / 2)
            y2 = int(cy + side / 2)

            line.append((x1, y1, x2, y2))

        grid.append(line)

    return grid, candidates


def compute_check_ratios(img_bin, grid):
    ratios = np.zeros((N_ROWS, N_COLS), dtype=float)

    h_img, w_img = img_bin.shape

    for row in range(N_ROWS):
        for col in range(N_COLS):
            x1, y1, x2, y2 = grid[row][col]

            x1 = max(0, min(w_img - 1, x1))
            x2 = max(0, min(w_img, x2))
            y1 = max(0, min(h_img - 1, y1))
            y2 = max(0, min(h_img, y2))

            w = x2 - x1
            h = y2 - y1

            if w <= 2 or h <= 2:
                ratios[row, col] = 0.0
                continue

            pad_x = int(INNER_PAD_RATIO * w)
            pad_y = int(INNER_PAD_RATIO * h)

            inner = img_bin[
                y1 + pad_y:y2 - pad_y,
                x1 + pad_x:x2 - pad_x
            ]

            if inner.size == 0:
                ratios[row, col] = 0.0
            else:
                ratios[row, col] = inner.sum() / inner.size

    return ratios


def read_student_id_grid_from_path(path):
    img_gray = load_gray_image(path)
    img_bin = binarize_dark_pixels_for_id(img_gray)

    grid, candidates = reconstruct_id_grid(img_bin)

    if grid is None:
        return {
            "student_id": None,
            "status": "grid_not_found",
            "grid": None,
            "ratios": None,
            "candidates": candidates
        }

    ratios = compute_check_ratios(img_bin, grid)

    digits = []
    ambiguous = False

    for col in range(N_COLS):
        col_ratios = ratios[:, col]
        best_row = int(np.argmax(col_ratios))
        best_ratio = float(col_ratios[best_row])

        if best_ratio < MIN_CHECK_RATIO:
            ambiguous = True
            digits.append("?")
        else:
            digits.append(str(best_row))

    if ambiguous:
        student_id = None
        status = "not_enough_ink_in_one_or_more_columns"
    else:
        student_id = "".join(digits)
        status = "ok"

    return {
        "student_id": student_id,
        "status": status,
        "grid": grid,
        "ratios": ratios,
        "candidates": candidates
    }


# ============================================================
# DEBUG VISUEL
# ============================================================

def draw_rect_rgb(rgb, bbox, color_value):
    x1, y1, x2, y2 = bbox

    rr, cc = draw.rectangle_perimeter(
        start=(y1, x1),
        end=(y2, x2),
        shape=rgb.shape
    )

    rgb[rr, cc, 0] = color_value[0]
    rgb[rr, cc, 1] = color_value[1]
    rgb[rr, cc, 2] = color_value[2]


def draw_results(img_gray, signature_box, grid_info, out_path):
    base = img_gray.astype(float)
    base = exposure.rescale_intensity(base, out_range=(0, 1))

    rgb = np.dstack([base, base, base])

    if signature_box is not None:
        draw_rect_rgb(rgb, signature_box, (0.0, 0.3, 1.0))

    if grid_info is not None and grid_info.get("grid") is not None:
        grid = grid_info["grid"]
        ratios = grid_info.get("ratios")

        chosen_rows = []

        if ratios is not None:
            for col in range(N_COLS):
                chosen_rows.append(int(np.argmax(ratios[:, col])))

        for row in range(N_ROWS):
            for col in range(N_COLS):
                bbox = grid[row][col]

                if col < len(chosen_rows) and row == chosen_rows[col]:
                    draw_rect_rgb(rgb, bbox, (0.0, 1.0, 0.0))
                else:
                    draw_rect_rgb(rgb, bbox, (1.0, 0.0, 0.0))

    io.imsave(out_path, (rgb * 255).astype(np.uint8))


def save_sanitized_signature_debug(clean_binary, out_path):
    img = (clean_binary * 255).astype(np.uint8)
    io.imsave(out_path, img)


# ============================================================
# MAIN
# ============================================================

def filename_contains_student_id(filename, student_id):
    return student_id is not None and filename.find(student_id) != -1


def main():
    ensure_clean_dir(SIGNATURE_FAILS_DIR)
    ensure_clean_dir(ID_FAILS_DIR)

    if SAVE_OK_DEBUG:
        ensure_clean_dir(DEBUG_OK_DIR)

    if not isdir(SOURCE_PATH_DATA):
        print(f"Dossier introuvable : {SOURCE_PATH_DATA}")
        return

    print("Chargement du CNN signature...")
    signature_model, label_index_to_student_id = load_signature_model_and_labels()
    print("CNN chargé.")
    print("Nombre de classes CNN :", len(label_index_to_student_id))

    source_dir = listdir(SOURCE_PATH_DATA)
    source_dir = [form for form in source_dir if re.match(r"^FORM\d$", form)]
    source_dir.sort()

    list_id_valides = []
    list_id_non_valides = []

    list_signature_valide = []
    list_signature_non_valide = []

    csv_rows = []

    for form in source_dir:
        temp_dir_path = join(SOURCE_PATH_DATA, form, f"EXAM_{form}_PRESENCES")

        if not isdir(temp_dir_path):
            print("Dossier présence introuvable :", temp_dir_path)
            continue

        presence_pages = sorted(listdir(temp_dir_path))

        for current_presence_page in presence_pages:
            if not current_presence_page.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue

            current_presence_page_path = join(temp_dir_path, current_presence_page)

            print("\nTraitement :", current_presence_page_path)

            try:
                img_gray = load_gray_image(current_presence_page_path)
                img_binaire = load_binary_image(current_presence_page_path)
            except Exception as e:
                print("Erreur lecture image :", e)

                csv_rows.append([
                    form,
                    current_presence_page,
                    "",
                    "",
                    "",
                    "image_error",
                    "",
                    ""
                ])

                continue

            # ------------------------------------------------
            # 1) Signature box avec ancienne méthode
            # ------------------------------------------------
            item_regions = find_regions(img_binaire)
            signature_box = find_signature_box(item_regions, img_binaire.shape)

            has_signature, ink_ratio = detect_if_the_signature_is_there(
                img_binaire,
                signature_box,
                SIGNATURE_MIN_INK_RATIO
            )

            # ------------------------------------------------
            # 2) Student ID par grille
            # ------------------------------------------------
            id_result = read_student_id_grid_from_path(current_presence_page_path)
            student_id_grid = id_result["student_id"]

            grid_info = {
                "grid": id_result.get("grid"),
                "ratios": id_result.get("ratios"),
                "status": id_result.get("status")
            }

            if student_id_grid is None:
                list_id_non_valides.append((
                    form,
                    current_presence_page,
                    id_result["status"]
                ))

                draw_results(
                    img_gray,
                    signature_box,
                    grid_info,
                    join(ID_FAILS_DIR, current_presence_page)
                )

                id_valid = False

            elif filename_contains_student_id(current_presence_page, student_id_grid):
                list_id_valides.append((form, current_presence_page, student_id_grid))
                id_valid = True

            else:
                list_id_non_valides.append((
                    form,
                    current_presence_page,
                    student_id_grid,
                    "id_ne_correspond_pas_au_nom"
                ))

                draw_results(
                    img_gray,
                    signature_box,
                    grid_info,
                    join(ID_FAILS_DIR, current_presence_page)
                )

                id_valid = False

            # ------------------------------------------------
            # 3) Signature CNN
            # ------------------------------------------------
            student_id_signature = None
            signature_confidence = 0.0
            signature_status = ""

            if signature_box is None:
                signature_status = "signature_box_not_found"

            elif not has_signature:
                signature_status = "no_signature_ink"

            else:
                signature_crop = crop_signature_inside_box(img_gray, signature_box)

                if signature_crop is None:
                    signature_status = "signature_crop_empty"

                else:
                    try:
                        (
                            student_id_signature,
                            signature_confidence,
                            clean_binary_signature
                        ) = predict_signature_student_id(
                            signature_model,
                            label_index_to_student_id,
                            signature_crop
                        )

                        if student_id_signature is None:
                            signature_status = "cnn_low_confidence"

                            debug_name = current_presence_page.rsplit(".", 1)[0] + "_cnn_low_conf.png"

                            save_sanitized_signature_debug(
                                clean_binary_signature,
                                join(SIGNATURE_FAILS_DIR, debug_name)
                            )

                        else:
                            signature_status = "ok"

                    except Exception as e:
                        signature_status = f"cnn_error:{e}"

            # ------------------------------------------------
            # 4) Validation signature
            # ------------------------------------------------
            if student_id_signature is None:
                list_signature_non_valide.append((
                    form,
                    current_presence_page,
                    student_id_grid,
                    None,
                    signature_confidence,
                    signature_status
                ))

                draw_results(
                    img_gray,
                    signature_box,
                    grid_info,
                    join(SIGNATURE_FAILS_DIR, current_presence_page)
                )

                signature_valid = False

            elif student_id_grid is not None and student_id_signature == student_id_grid:
                list_signature_valide.append((
                    form,
                    current_presence_page,
                    student_id_signature,
                    signature_confidence
                ))

                signature_valid = True

            else:
                list_signature_non_valide.append((
                    form,
                    current_presence_page,
                    student_id_grid,
                    student_id_signature,
                    signature_confidence,
                    "signature_id_different_from_grid_id"
                ))

                draw_results(
                    img_gray,
                    signature_box,
                    grid_info,
                    join(SIGNATURE_FAILS_DIR, current_presence_page)
                )

                signature_valid = False

            # ------------------------------------------------
            # 5) CSV
            # ------------------------------------------------
            csv_rows.append([
                form,
                current_presence_page,
                student_id_grid if student_id_grid is not None else "",
                student_id_signature if student_id_signature is not None else "",
                f"{signature_confidence:.4f}",
                id_result["status"],
                signature_status,
                f"{ink_ratio:.6f}"
            ])

            print("ID grille       :", student_id_grid)
            print("ID signature    :", student_id_signature)
            print("Confiance CNN   :", f"{signature_confidence:.4f}")
            print("Status ID       :", id_result["status"])
            print("Status signature:", signature_status)
            print("ID valide       :", id_valid)
            print("Signature valide:", signature_valid)

            if SAVE_OK_DEBUG and id_valid and signature_valid:
                draw_results(
                    img_gray,
                    signature_box,
                    grid_info,
                    join(DEBUG_OK_DIR, current_presence_page)
                )

    # ========================================================
    # SAUVEGARDE CSV
    # ========================================================

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "form",
            "imageName",
            "studentID_grid",
            "studentID_signature",
            "signature_confidence",
            "id_status",
            "signature_status",
            "signature_ink_ratio"
        ])

        writer.writerows(csv_rows)

    print("\n================ RÉSUMÉ ================")
    print("nb ID grille validés :", len(list_id_valides))
    print("nb ID grille non validés :", len(list_id_non_valides))

    print("nb signatures CNN validées :", len(list_signature_valide))
    print("nb signatures CNN non validées :", len(list_signature_non_valide))

    print("\nCSV sauvegardé :", OUTPUT_CSV)

    print("\nListe signatures CNN valides :")
    print(list_signature_valide)

    print("\nListe signatures CNN non valides :")
    print(list_signature_non_valide)


if __name__ == "__main__":
    main()