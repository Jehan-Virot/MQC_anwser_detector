# student_id_grid_detector.py
# Programme autonome pour lire le Student ID dans la grille 5 x 10.
# Méthodes utilisées : niveaux de gris, étirement d'histogramme, seuillage,
# morphologie simple, étiquetage de régions, profils / regroupement de coordonnées.

import os
import re
import csv
import itertools
import numpy as np
from skimage import io, color, transform, exposure, filters, measure, morphology, draw


# -----------------------------
# Paramètres principaux
# -----------------------------
SOURCE_PATH_DATA = "data/"
OUTPUT_CSV = "student_id_grid_results.csv"
DEBUG_DIR = "id_debug"

# ROI approximative de la grille STUDENT ID dans la page redressée.
# Format : x_min, y_min, x_max, y_max en pourcentage de largeur / hauteur.
ID_ROI_RATIO = (0.68, 0.12, 0.98, 0.55)

N_COLS = 5
N_ROWS = 10

# Seuil minimum pour dire qu'une case est cochée.
# Si les images sont très propres, 0.035 suffit souvent.
# Si les X sont très fins/faibles, descendre vers 0.025.
MIN_CHECK_RATIO = 0.030

# Padding interne : on enlève les bords imprimés de la case.
INNER_PAD_RATIO = 0.27


# -----------------------------
# Chargement et prétraitement
# -----------------------------
def load_gray_image(path):
    """Charge une image, la convertit en niveaux de gris float [0,1], et la met en portrait."""
    img = io.imread(path)

    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img.astype(float)
        if img_gray.max() > 1.0:
            img_gray = img_gray / 255.0

    # Même logique que ton turn_image : si la photo est en paysage, on tourne.
    H, W = img_gray.shape
    if W > H:
        img_gray = transform.rotate(img_gray, angle=270, resize=True, preserve_range=True)

    return img_gray


def stretch_histo(img_gray, p_low=5, p_high=95):
    """Etirement d'histogramme basique."""
    low, high = np.percentile(img_gray, (p_low, p_high))
    if high <= low:
        return img_gray
    return exposure.rescale_intensity(img_gray, in_range=(low, high), out_range=(0, 1))


def binarize_dark_pixels(img_gray):
    """
    Binarise les pixels foncés.
    Résultat : True = noir / encre / traits imprimés.
    """
    img_stretched = stretch_histo(img_gray)
    otsu = filters.threshold_otsu(img_stretched)

    # On rend le seuil un peu plus strict que Otsu pour éviter les ombres gris clair.
    # 0.92 garde les traits noirs et l'encre, mais rejette mieux le fond.
    threshold = otsu * 0.92
    img_bin = img_stretched < threshold

    # Nettoyage très léger. Pas de grosse érosion ici, sinon les X manuscrits fins disparaissent.
    img_bin = morphology.remove_small_objects(img_bin, min_size=8)
    img_bin = morphology.binary_closing(img_bin, morphology.square(2))

    return img_bin, img_stretched


# -----------------------------
# Détection de la grille 5 x 10
# -----------------------------
def crop_id_roi(img):
    """Retourne la sous-image où se trouve normalement la grille STUDENT ID."""
    H, W = img.shape[:2]
    rx1, ry1, rx2, ry2 = ID_ROI_RATIO
    x1 = int(rx1 * W)
    y1 = int(ry1 * H)
    x2 = int(rx2 * W)
    y2 = int(ry2 * H)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def find_square_candidates(img_bin):
    """
    Trouve des candidats de cases carrées dans la ROI.
    Les coordonnées retournées sont relatives à la ROI.
    """
    roi, _ = crop_id_roi(img_bin)

    # Une petite fermeture aide quand un bord de case est coupé.
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

        # Bornes relatives, pour rester valables si la résolution change.
        min_side = max(10, int(0.015 * H))
        max_side = max(25, int(0.110 * H))

        ratio = w / h if h > 0 else 0
        extent = area / float(w * h) if w * h > 0 else 0

        # Une case vide est surtout un contour : extent faible à moyen.
        # Une case cochée a plus d'encre, mais reste dans la même bbox.
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
    """Regroupe des coordonnées 1D proches, puis retourne les centres des groupes."""
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
    """
    Sélectionne n_expected groupes dont les centres forment une grille régulière.
    C'est plus robuste que juste prendre les plus gros composants.
    """
    if len(clusters) < n_expected:
        return None

    clusters = sorted(clusters, key=lambda c: c["center"])

    # Si trop de groupes parasites, on teste les combinaisons raisonnables.
    best_combo = None
    best_cost = float("inf")

    max_count = max(c["count"] for c in clusters)
    all_combos = itertools.combinations(clusters, n_expected)

    for combo in all_combos:
        centers = np.array([c["center"] for c in combo], dtype=float)
        gaps = np.diff(centers)
        if np.any(gaps <= 0):
            continue

        mean_gap = np.mean(gaps)
        if mean_gap <= 0:
            continue

        regularity_cost = np.std(gaps) / mean_gap
        count_bonus = sum(c["count"] for c in combo) / float(n_expected * max_count)

        # On veut une grille régulière et beaucoup de cases détectées par ligne/colonne.
        cost = regularity_cost - 0.10 * count_bonus

        if cost < best_cost:
            best_cost = cost
            best_combo = combo

    if best_combo is None:
        return None

    return sorted([float(c["center"]) for c in best_combo])


def reconstruct_grid(img_bin):
    """
    Reconstruit la grille Student ID.
    Retourne une liste 10 x 5 de bboxes absolues : grid[row][col] = (x1,y1,x2,y2).
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

    # Side légèrement agrandi, pour que la bbox recouvre bien la case même si le composant est incomplet.
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


# -----------------------------
# Lecture des cases cochées
# -----------------------------
def ink_ratio_inside_box(img_bin, bbox, pad_ratio=INNER_PAD_RATIO):
    """Calcule le ratio de pixels noirs dans l'intérieur de la case, sans compter le contour."""
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
    """
    Lit le Student ID : 5 colonnes, 10 lignes.
    Chaque colonne donne un chiffre, la ligne cochée donne la valeur 0..9.
    """
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

        # Seuil dynamique : utile quand certaines images sont plus bruitées.
        dynamic_threshold = max(MIN_CHECK_RATIO, float(np.mean(col_ratios) + 1.6 * np.std(col_ratios)))

        # On accepte si la meilleure case est assez noire et plus forte que les autres.
        if best >= dynamic_threshold and best >= second + 0.010:
            digits.append(str(best_row))
            confidence.append(best - second)
        else:
            digits.append("?")
            confidence.append(best - second)

    return "".join(digits), ratios, confidence


def detect_student_id(path):
    """Fonction principale pour une image."""
    img_gray = load_gray_image(path)
    img_bin, img_stretched = binarize_dark_pixels(img_gray)

    grid, candidates = reconstruct_grid(img_bin)
    if grid is None:
        return {
            "student_id": "?????",
            "ok": False,
            "reason": "grille_non_detectee",
            "grid": None,
            "ratios": None,
            "img_bin": img_bin,
            "img_gray": img_gray,
            "candidates": candidates,
        }

    student_id, ratios, confidence = read_student_id_from_grid(img_bin, grid)
    ok = "?" not in student_id

    return {
        "student_id": student_id,
        "ok": ok,
        "reason": "ok" if ok else "case_ambigue",
        "grid": grid,
        "ratios": ratios,
        "confidence": confidence,
        "img_bin": img_bin,
        "img_gray": img_gray,
        "candidates": candidates,
    }


# -----------------------------
# Debug visuel
# -----------------------------
def draw_debug_image(result, out_path):
    """Sauvegarde une image avec la grille détectée et les cases choisies."""
    img = result["img_gray"]
    rgb = np.dstack([img, img, img])
    rgb = exposure.rescale_intensity(rgb, out_range=(0, 1))

    grid = result["grid"]
    ratios = result["ratios"]

    if grid is not None:
        chosen_rows = []
        if ratios is not None:
            for col in range(N_COLS):
                chosen_rows.append(int(np.argmax(ratios[:, col])))

        for row in range(N_ROWS):
            for col in range(N_COLS):
                x1, y1, x2, y2 = grid[row][col]
                rr, cc = draw.rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=rgb.shape)

                # Rouge = case, vert = case lue comme cochée.
                if col < len(chosen_rows) and row == chosen_rows[col]:
                    rgb[rr, cc, 0] = 0.0
                    rgb[rr, cc, 1] = 1.0
                    rgb[rr, cc, 2] = 0.0
                else:
                    rgb[rr, cc, 0] = 1.0
                    rgb[rr, cc, 1] = 0.0
                    rgb[rr, cc, 2] = 0.0

    io.imsave(out_path, (rgb * 255).astype(np.uint8))


# -----------------------------
# Traitement dossier projet
# -----------------------------
def iter_presence_images(source_path_data=SOURCE_PATH_DATA):
    """Parcourt data/FORMx/EXAM_FORMx_PRESENCES/ comme dans ton code."""
    if not os.path.isdir(source_path_data):
        return

    forms = [d for d in os.listdir(source_path_data) if re.match(r"^FORM\d+$", d)]
    forms.sort()

    for form in forms:
        presence_dir = os.path.join(source_path_data, form, f"EXAM_{form}_PRESENCES")
        if not os.path.isdir(presence_dir):
            # Variante possible selon les noms de dossiers.
            presence_dir = os.path.join(source_path_data, form, f"EXAM_{form}_PRESENCES/")

        if not os.path.isdir(presence_dir):
            continue

        for name in sorted(os.listdir(presence_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                yield form, name, os.path.join(presence_dir, name)


def process_all_images(source_path_data=SOURCE_PATH_DATA, output_csv=OUTPUT_CSV, debug=True):
    if debug and not os.path.exists(DEBUG_DIR):
        os.mkdir(DEBUG_DIR)

    rows = []

    for form, name, path in iter_presence_images(source_path_data):
        print("Traitement :", path)
        try:
            result = detect_student_id(path)
        except Exception as e:
            print("  ERREUR :", e)
            rows.append([name, "?????", "erreur"])
            continue

        student_id = result["student_id"]
        print("  Student ID lu :", student_id, "|", result["reason"])
        rows.append([name, student_id, result["reason"]])

        if debug:
            base = os.path.splitext(name)[0]
            out_path = os.path.join(DEBUG_DIR, f"{form}_{base}_debug.jpg")
            draw_debug_image(result, out_path)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["imageName", "studentID_grid", "status"])
        writer.writerows(rows)

    print("\nRésultats sauvegardés dans", output_csv)
    if debug:
        print("Images de debug sauvegardées dans", DEBUG_DIR)


# -----------------------------
# Lancement
# -----------------------------
if __name__ == "__main__":
    # Test rapide sur une seule image :
    # python student_id_grid_detector.py chemin/vers/image.jpg
    import sys

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        result = detect_student_id(image_path)
        print("Student ID lu :", result["student_id"])
        print("Status :", result["reason"])
        if not os.path.exists(DEBUG_DIR):
            os.mkdir(DEBUG_DIR)
        debug_path = os.path.join(DEBUG_DIR, "single_debug.jpg")
        draw_debug_image(result, debug_path)
        print("Debug :", debug_path)
    else:
        # Sinon, traite tout le dossier data/FORMx/EXAM_FORMx_PRESENCES/
        process_all_images(SOURCE_PATH_DATA, OUTPUT_CSV, debug=True)
