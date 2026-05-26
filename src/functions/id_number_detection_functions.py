import numpy as np
from functions.detect_inside_boxes import ink_ratio_inside_box

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

def find_student_id_grid_v2(square_boxes, image_shape):
    H, W = image_shape

    candidates = [
        b for b in square_boxes
        if b["cx"] > 0.60 * W and b["cy"] < 0.55 * H
    ]

    if len(candidates) < 25:
        return None, None, None

    x_centers = cluster_1d_fixed([b["cx"] for b in candidates], k=5)
    y_centers = cluster_1d_fixed([b["cy"] for b in candidates], k=10)

    if x_centers is None or y_centers is None:
        return None, None, None

    return candidates, x_centers, y_centers

def cluster_1d_fixed(values, k, max_iter=50):
    values = np.array(sorted(values), dtype=float)

    if len(values) < k:
        return None

    # Initialisation par percentiles
    centers = np.percentile(values, np.linspace(0, 100, k))

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        new_centers = []
        for i in range(k):
            group = values[labels == i]
            if len(group) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(np.mean(group))

        new_centers = np.array(new_centers)

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return sorted(centers.tolist())

def nearest_box(boxes, x, y):
    best_box = None
    best_dist = float("inf")

    for b in boxes:
        d = (b["cx"] - x) ** 2 + (b["cy"] - y) ** 2

        if d < best_dist:
            best_dist = d
            best_box = b

    return best_box

def read_student_id_v2(img_binaire, square_boxes, image_shape, min_ratio=1.8):
    candidates, x_centers, y_centers = find_student_id_grid_v2(square_boxes, image_shape)

    if candidates is None:
        return None, None

    matrix = np.zeros((10, 5))

    for row, y in enumerate(y_centers):
        for col, x in enumerate(x_centers):
            b = nearest_box(candidates, x, y)

            # Sécurité : ne pas accepter une boîte trop loin du centre attendu
            dist = ((b["cx"] - x) ** 2 + (b["cy"] - y) ** 2) ** 0.5
            if dist > 18:
                matrix[row, col] = 0
            else:
                matrix[row, col] = ink_ratio_inside_box(img_binaire, b["bbox"])

    digits = []

    for col in range(5):
        column_values = matrix[:, col]

        best_row = int(np.argmax(column_values))
        best_value = column_values[best_row]
        median_value = np.median(column_values)

        # Une case cochée doit être clairement plus remplie que les autres
        if best_value > max(0.08, median_value * min_ratio):
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