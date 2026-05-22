from skimage import io, color, measure
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import re
# =========================
# Paramètres
# =========================


DARK_THRESHOLD = 0.45
MIN_CHECKED_EXTRA_RATIO = 1.20
SIGNATURE_MIN_INK_RATIO = 0.006


# =========================
# Chargement et binarisation
# =========================

def load_binary_image(path):
    img = io.imread(path)

    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img / 255.0

    binary = gray < DARK_THRESHOLD

    return img, gray, binary


# =========================
# Composantes connexes
# =========================

def find_components(binary):
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    return props


# =========================
# Détection du bloc signature
# =========================

def find_signature_box(props, image_shape):
    H, W = image_shape
    candidates = []

    for p in props:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        is_large_enough = w > 0.20 * W and h > 0.08 * H
        is_not_too_large = w < 0.50 * W and h < 0.25 * H
        is_left = minc < 0.50 * W
        is_upper = minr < 0.55 * H
        good_ratio = 1.2 < (w / h) < 3.5

        if is_large_enough and is_not_too_large and is_left and is_upper and good_ratio:
            candidates.append((area, minc, minr, maxc, maxr))

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, reverse=True)
    _, x1, y1, x2, y2 = candidates[0]

    return x1, y1, x2, y2


def signature_is_present(binary, signature_box):
    if signature_box is None:
        return False, 0.0

    x1, y1, x2, y2 = signature_box

    w = x2 - x1
    h = y2 - y1

    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    inside = binary[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

    ink_ratio = inside.sum() / inside.size
    present = ink_ratio > SIGNATURE_MIN_INK_RATIO

    return present, ink_ratio


# =========================
# Détection des cases carrées
# =========================

def find_square_boxes(props, image_shape):
    boxes = []

    for p in props:
        minr, minc, maxr, maxc = p.bbox

        h = maxr - minr
        w = maxc - minc
        area = p.area

        if 20 <= w <= 45 and 20 <= h <= 45:
            ratio = w / h

            if 0.7 <= ratio <= 1.3 and area >= 180:
                boxes.append({
                    "bbox": (minc, minr, maxc, maxr),
                    "cx": (minc + maxc) / 2,
                    "cy": (minr + maxr) / 2,
                    "area": area,
                    "w": w,
                    "h": h
                })

    return boxes


def group_positions(values, max_gap):
    values = sorted(values)

    if len(values) == 0:
        return []

    groups = []
    current = [values[0]]

    for v in values[1:]:
        if abs(v - current[-1]) <= max_gap:
            current.append(v)
        else:
            groups.append(current)
            current = [v]

    groups.append(current)

    centers = [float(np.mean(g)) for g in groups]
    return centers


# =========================
# Détection de la grille Student ID
# =========================

def find_student_id_grid(square_boxes, image_shape):
    H, W = image_shape

    candidates = [
        b for b in square_boxes
        if b["cx"] > 0.60 * W and b["cy"] < 0.55 * H
    ]

    x_centers = group_positions([b["cx"] for b in candidates], max_gap=22)
    y_centers = group_positions([b["cy"] for b in candidates], max_gap=25)

    x_centers = sorted(x_centers)
    y_centers = sorted(y_centers)

    # On garde les 5 colonnes de la grille Student ID
    if len(x_centers) > 5:
        x_centers = x_centers[-5:]

    # Correction importante :
    # Si une ligne parasite est détectée au-dessus des chiffres 0-9,
    # on la supprime.
    if len(y_centers) == 11:
        y_centers = y_centers[1:11]

    # Sécurité si plus de 11 lignes sont trouvées
    elif len(y_centers) > 11:
        y_centers = y_centers[-10:]

    # Cas normal
    elif len(y_centers) > 10:
        y_centers = y_centers[1:11]

    if len(x_centers) != 5 or len(y_centers) != 10:
        return None, None, None

    return candidates, x_centers, y_centers


def nearest_box(boxes, x, y):
    best_box = None
    best_dist = float("inf")

    for b in boxes:
        d = (b["cx"] - x) ** 2 + (b["cy"] - y) ** 2

        if d < best_dist:
            best_dist = d
            best_box = b

    return best_box


def read_student_id(square_boxes, image_shape):
    candidates, x_centers, y_centers = find_student_id_grid(square_boxes, image_shape)

    if candidates is None:
        return None, None

    matrix = np.zeros((10, 5))

    for row, y in enumerate(y_centers):
        for col, x in enumerate(x_centers):
            b = nearest_box(candidates, x, y)
            matrix[row, col] = b["area"]

    digits = []

    for col in range(5):
        column_values = matrix[:, col]

        best_row = int(np.argmax(column_values))
        best_value = column_values[best_row]
        median_value = np.median(column_values)

        if best_value > median_value * MIN_CHECKED_EXTRA_RATIO:
            digits.append(str(best_row))
        else:
            digits.append("?")

    student_id = "".join(digits)

    grid_info = {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "matrix": matrix
    }

    return student_id, grid_info


# =========================
# Affichage / sauvegarde
# =========================

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

        for row, y in enumerate(y_centers):
            for col, x in enumerate(x_centers):
                ax.plot(x, y, marker="o", markersize=4)

            ax.text(x_centers[0] - 40, y, str(row), fontsize=10)

        ax.text(x_centers[0], y_centers[0] - 30, "Grille Student ID", fontsize=12)

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print("Image sauvegardée :", output_path)


# =========================
# Programme principal
# =========================

SOURCE_PATH_DATA = "data/"

source_dir = listdir(SOURCE_PATH_DATA)
source_dir = [form for form in source_dir if re.match(r"^FORM\d$", form)]
list_etudiant_valide = []
list_etudiant_non_valide = []
list_etudiant_signature = []
list_etudiant_non_signature = []
for form in source_dir:
    temp_dir_path = f"{SOURCE_PATH_DATA}{form}/EXAM_{form}_PRESENCES/"
    presence_pages = listdir(temp_dir_path)
    for current_presence_page in presence_pages:
        current_presence_page_path = f"{temp_dir_path}{current_presence_page}"
        
        print(current_presence_page_path)
        img, gray, binary = load_binary_image(current_presence_page_path)

        props = find_components(binary)

        signature_box = find_signature_box(props, binary.shape)
        has_signature, ink_ratio = signature_is_present(binary, signature_box)

        square_boxes = find_square_boxes(props, binary.shape)
        student_id, grid_info = read_student_id(square_boxes, binary.shape)

        
        if student_id == None:
            list_etudiant_non_valide.append(student_id)
        elif current_presence_page.find(student_id) != -1:
            list_etudiant_valide.append(student_id)
        else:
            list_etudiant_non_valide.append(student_id)
        if has_signature:
            list_etudiant_signature.append(student_id)
        else:
            list_etudiant_non_signature.append(student_id)
            
print(f"nb étudiant avec ID détécté : {len(list_etudiant_valide)}/{len(list_etudiant_valide)+len(list_etudiant_non_valide)}")
print(f"nb étudiant avec ID non détécté : {len(list_etudiant_non_valide)}/{len(list_etudiant_valide)+len(list_etudiant_non_valide)}")

print(f"nb étudiant avec signature détécté : {len(list_etudiant_signature)}/{len(list_etudiant_signature)+len(list_etudiant_non_signature)}")
print(f"nb étudiant avec signature non détécté : {len(list_etudiant_non_signature)}/{len(list_etudiant_signature)+len(list_etudiant_non_signature)}")

print("liste étudiant id detect : ", list_etudiant_valide)
print("liste étudiant id detect : ", list_etudiant_non_valide)
print("liste étudiant signature detect : ", list_etudiant_signature)
print("liste étudiant signature detect : ", list_etudiant_non_signature)
