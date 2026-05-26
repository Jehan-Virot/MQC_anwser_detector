# main_merged_student_id_signature.py
# Un seul fichier pour :
# 1) lire le Student ID via la grille 5 x 10
# 2) détecter la présence d'une signature
# 3) parcourir data/FORMx/EXAM_FORMx_PRESENCES/
#
# Méthodes utilisées : niveaux de gris, étirement d'histogramme, seuillage Otsu,
# morphologie simple, labelling, regroupement de coordonnées / profils.

from os import listdir, mkdir
from os.path import exists, isdir, join, splitext
from shutil import rmtree
import re
import csv
import itertools
import numpy as np
from skimage import io, color, transform, exposure, filters, measure, morphology, draw


# ============================================================
# PARAMETRES
# ============================================================
SOURCE_PATH_DATA = "data/"

SIGNATURE_FAILS_DIR = "signature_fails"
ID_FAILS_DIR = "id_fails"
DEBUG_OK_DIR = "id_debug_ok"       # mets SAVE_OK_DEBUG = False si tu n'en veux pas
OUTPUT_CSV = "presence_results.csv"

SAVE_OK_DEBUG = False
SIGNATURE_MIN_INK_RATIO = 0.006

# Zone approximative de la grille STUDENT ID dans une page portrait.
# x_min, y_min, x_max, y_max en pourcentage de largeur / hauteur.
ID_ROI_RATIO = (0.68, 0.12, 0.98, 0.55)

N_COLS = 5
N_ROWS = 10

# Seuil minimum pour considérer une case comme cochée.
# Si tes croix sont fines/faibles : descendre vers 0.020-0.025.
# Si le fond gris est souvent pris comme encre : monter vers 0.035-0.045.
MIN_CHECK_RATIO = 0.030

# Pour ne pas compter le contour imprimé de chaque case.
INNER_PAD_RATIO = 0.27


# ============================================================
# OUTILS GENERAUX IMAGE
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
    H, W = img.shape[:2]
    if W > H:
        img = transform.rotate(img, angle=270, resize=True, preserve_range=True)
    return img


def stretch_histo(img_gray, p_low=5, p_high=95):
    low, high = np.percentile(img_gray, (p_low, p_high))
    if high <= low:
        return img_gray
    return exposure.rescale_intensity(img_gray, in_range=(low, high), out_range=(0, 1))


def load_gray_image(path):
    img = io.imread(path)
    img_gray = rgb_to_gray_and_normalize(img)
    img_gray = turn_image_if_needed(img_gray)
    return img_gray


def load_binary_image(path):
    """
    Version intégrée de ton chargement :
    image -> gris -> stretching -> Otsu -> True pour les pixels noirs.
    """
    img_gray = load_gray_image(path)
    img_stretched = stretch_histo(img_gray)
    threshold = filters.threshold_otsu(img_stretched)
    img_binaire = img_stretched < threshold
    return img_binaire


def binarize_dark_pixels_for_id(img_gray):
    """
    Binarisation un peu plus stricte que Otsu pour la grille ID.
    Cela limite les ombres et le gris de fond.
    """
    img_stretched = stretch_histo(img_gray)
    otsu = filters.threshold_otsu(img_stretched)
    threshold = otsu * 0.92

    img_bin = img_stretched < threshold
    img_bin = morphology.remove_small_objects(img_bin, min_size=8)
    img_bin = morphology.binary_closing(img_bin, morphology.square(2))
    return img_bin


# ============================================================
# SIGNATURE
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


def detect_if_the_signature_is_there(img_binaire, signature_box, ratio_pixel=SIGNATURE_MIN_INK_RATIO):
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


# ============================================================
# DETECTION STUDENT ID : GRILLE 5 x 10
# ============================================================
def crop_id_roi(img):
    H, W = img.shape[:2]
    rx1, ry1, rx2, ry2 = ID_ROI_RATIO
    x1 = int(rx1 * W)
    y1 = int(ry1 * H)
    x2 = int(rx2 * W)
    y2 = int(ry2 * H)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def find_square_candidates(img_bin):
    """Trouve des candidats de cases carrées dans la ROI Student ID."""
    roi, _ = crop_id_roi(img_bin)
    roi_clean = morphology.binary_closing(roi, morphology.square(2))

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


def reconstruct_id_grid(img_bin):
    """
    Retourne une grille 10 x 5 de bounding boxes absolues :
    grid[row][col] = (x1, y1, x2, y2)
    row correspond au chiffre 0..9, col à la position dans le Student ID.
    """
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

    for y in y_centers:
        row = []
        for x in x_centers:
            x1 = int(round(rx1 + x - side / 2))
            y1 = int(round(ry1 + y - side / 2))
            x2 = int(round(rx1 + x + side / 2))
            y2 = int(round(ry1 + y + side / 2))
            row.append((x1, y1, x2, y2))
        grid.append(row)

    return grid, candidates


def ink_ratio_inside_box(img_bin, bbox, pad_ratio=INNER_PAD_RATIO):
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
    ratios = np.zeros((N_ROWS, N_COLS), dtype=float)

    for row in range(N_ROWS):
        for col in range(N_COLS):
            ratios[row, col] = ink_ratio_inside_box(img_bin, grid[row][col])

    digits = []
    confidence = []

    for col in range(N_COLS):
        col_ratios = ratios[:, col]
        best_row = int(np.argmax(col_ratios))
        sorted_ratios = np.sort(col_ratios)
        best = float(sorted_ratios[-1])
        second = float(sorted_ratios[-2]) if len(sorted_ratios) >= 2 else 0.0

        dynamic_threshold = max(
            MIN_CHECK_RATIO,
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
    img_bin = binarize_dark_pixels_for_id(img_gray)

    grid, candidates = reconstruct_id_grid(img_bin)
    if grid is None:
        return {
            "student_id": None,
            "status": "grille_non_detectee",
            "grid": None,
            "ratios": None,
            "img_gray": img_gray,
            "img_bin": img_bin,
            "candidates": candidates,
        }

    student_id, ratios, confidence = read_student_id_from_grid(img_bin, grid)
    if "?" in student_id:
        student_id_out = None
        status = "case_ambigue_" + student_id
    else:
        student_id_out = student_id
        status = "ok"

    return {
        "student_id": student_id_out,
        "raw_student_id": student_id,
        "status": status,
        "grid": grid,
        "ratios": ratios,
        "confidence": confidence,
        "img_gray": img_gray,
        "img_bin": img_bin,
        "candidates": candidates,
    }


# ============================================================
# DEBUG VISUEL
# ============================================================
def draw_rect_rgb(rgb, bbox, color_value):
    x1, y1, x2, y2 = bbox
    rr, cc = draw.rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=rgb.shape)
    rgb[rr, cc, 0] = color_value[0]
    rgb[rr, cc, 1] = color_value[1]
    rgb[rr, cc, 2] = color_value[2]


def draw_results(img_gray, signature_box, grid_info, out_path):
    """
    Image debug :
    - bleu : boîte signature
    - rouge : cases ID reconstruites
    - vert : cases choisies pour former l'ID
    """
    if img_gray.dtype == bool:
        base = img_gray.astype(float)
    else:
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


# ============================================================
# MAIN MERGE
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

    source_dir = listdir(SOURCE_PATH_DATA)
    source_dir = [form for form in source_dir if re.match(r"^FORM\d$", form)]
    source_dir.sort()

    list_etudiant_valide = []
    list_etudiant_non_valide = []
    list_etudiant_signature = []
    list_etudiant_non_signature = []

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
            print(current_presence_page_path)

            try:
                img_gray = load_gray_image(current_presence_page_path)
                img_binaire = load_binary_image(current_presence_page_path)
            except Exception as e:
                print("format photo bug :", e)
                csv_rows.append([current_presence_page, "", "image_error", "", "image_error", ""])
                continue

            # Signature
            items_regions = find_regions(img_binaire)
            signature_box = find_signature_box(items_regions, img_binaire.shape)
            has_signature, ink_ratio = detect_if_the_signature_is_there(
                img_binaire,
                signature_box,
                SIGNATURE_MIN_INK_RATIO
            )

            # Student ID nouvelle méthode grille 5 x 10
            id_result = read_student_id_grid_from_path(current_presence_page_path)
            student_id = id_result["student_id"]
            grid_info = {
                "grid": id_result.get("grid"),
                "ratios": id_result.get("ratios"),
                "status": id_result.get("status"),
            }

            # Validation ID avec le nom de fichier, comme dans ton main
            if student_id is None:
                list_etudiant_non_valide.append((student_id, current_presence_page, id_result["status"]))
                out_path = join(ID_FAILS_DIR, current_presence_page)
                draw_results(img_gray, signature_box, grid_info, out_path)
                id_valid = False
            elif filename_contains_student_id(current_presence_page, student_id):
                list_etudiant_valide.append(student_id)
                id_valid = True
            else:
                list_etudiant_non_valide.append((student_id, current_presence_page, "id_ne_correspond_pas_au_nom"))
                out_path = join(ID_FAILS_DIR, current_presence_page)
                draw_results(img_gray, signature_box, grid_info, out_path)
                id_valid = False

            # Validation signature
            if has_signature:
                list_etudiant_signature.append(student_id)
                signature_status = "signature_detectee"
            else:
                list_etudiant_non_signature.append((student_id, current_presence_page))
                out_path = join(SIGNATURE_FAILS_DIR, current_presence_page)
                draw_results(img_gray, signature_box, grid_info, out_path)
                signature_status = "signature_non_detectee"

            if SAVE_OK_DEBUG and id_valid and has_signature:
                base, ext = splitext(current_presence_page)
                out_path = join(DEBUG_OK_DIR, base + "_debug" + ext)
                draw_results(img_gray, signature_box, grid_info, out_path)

            csv_rows.append([
                current_presence_page,
                student_id if student_id is not None else "",
                "id_ok" if id_valid else id_result["status"],
                "yes" if has_signature else "no",
                signature_status,
                ink_ratio,
            ])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["imageName", "studentID_grid", "studentID_status", "has_signature", "signature_status", "signature_ink_ratio"])
        writer.writerows(csv_rows)

    total_id = len(list_etudiant_valide) + len(list_etudiant_non_valide)
    total_sig = len(list_etudiant_signature) + len(list_etudiant_non_signature)

    print(f"nb étudiant avec ID détécté : {len(list_etudiant_valide)}/{total_id}")
    print(f"nb étudiant avec ID non détécté : {len(list_etudiant_non_valide)}/{total_id}")

    print(f"nb étudiant avec signature détécté : {len(list_etudiant_signature)}/{total_sig}")
    print(f"nb étudiant avec signature non détécté : {len(list_etudiant_non_signature)}/{total_sig}")

    print("liste étudiant id detect : ", list_etudiant_valide)
    print("liste étudiant non id detect : ", list_etudiant_non_valide)
    print("liste étudiant signature detect : ", list_etudiant_signature)
    print("liste étudiant non signature detect : ", list_etudiant_non_signature)
    print("CSV sauvegardé :", OUTPUT_CSV)


if __name__ == "__main__":
    main()
