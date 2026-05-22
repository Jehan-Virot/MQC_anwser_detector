import numpy as np

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


def read_student_id(square_boxes, image_shape, min_ratio):
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

        if best_value > median_value * min_ratio:
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