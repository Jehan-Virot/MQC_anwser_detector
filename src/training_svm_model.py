# train_signature_svm.py
# Programme uniquement pour entraîner un SVM sur les signatures.
#
# Lancement :
#   python train_signature_svm.py
#
# Sorties :
#   signature_model_svm/svm_shape.joblib
#   signature_model_svm/labels.json

import os
import json
import random
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np

from skimage import io, color, filters, morphology, measure, transform, exposure
from scipy import ndimage as ndi

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PARAMÈTRES
# ============================================================

SOURCE_PATH = "data/STUDENT_CLASS_SIGNATURES"

OUTPUT_DIR = "signature_model_svm"
MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_shape.joblib")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.json")

RANDOM_SEED = 4564
TRAIN_RATIO = 0.80

MIN_OBJECT_SIZE = 8
PRUNING_ITERATIONS = 8

N_FOURIER_POINTS = 128
N_FOURIER_DESCRIPTORS = 16

AUGMENT_PER_IMAGE = 4


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def safe_div(a, b, eps=1e-8):
    return float(a) / float(b + eps)


def rgb_to_gray_and_normalize(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def stretch_histo(img_gray, p_low=2, p_high=98):
    low, high = np.percentile(img_gray, (p_low, p_high))

    if high <= low:
        return img_gray.astype(np.float32)

    return exposure.rescale_intensity(
        img_gray,
        in_range=(low, high),
        out_range=(0.0, 1.0)
    ).astype(np.float32)


# ============================================================
# PRÉTRAITEMENT SIGNATURE
# ============================================================

def count_8_neighbors(binary_img):
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=np.uint8
    )

    return ndi.convolve(
        binary_img.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0
    )


def prune_skeleton(skeleton, iterations=PRUNING_ITERATIONS):
    pruned = skeleton.astype(bool).copy()

    for _ in range(iterations):
        neighbors = count_8_neighbors(pruned)
        endpoints = pruned & (neighbors == 1)

        if not np.any(endpoints):
            break

        pruned[endpoints] = False

    return pruned


def crop_to_foreground(binary_img, pad=8):
    coords = np.argwhere(binary_img)

    if coords.size == 0:
        return binary_img

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1

    h, w = binary_img.shape[:2]

    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(h, y2 + pad)
    x2 = min(w, x2 + pad)

    return binary_img[y1:y2, x1:x2]


def preprocess_signature_binary_from_array(img):
    """
    Image signature -> image binaire.
    Convention :
        fond = False
        encre = True
    """
    img_gray = rgb_to_gray_and_normalize(img)
    img_gray = stretch_histo(img_gray)

    img_smooth = filters.gaussian(
        img_gray,
        sigma=1.0,
        preserve_range=True
    )

    try:
        threshold = filters.threshold_otsu(img_smooth)
    except ValueError:
        return np.zeros_like(img_smooth, dtype=bool)

    img_bin = img_smooth < threshold

    img_bin = morphology.remove_small_objects(
        img_bin.astype(bool),
        min_size=MIN_OBJECT_SIZE
    )

    img_bin = morphology.closing(
        img_bin,
        np.ones((2, 2), dtype=bool)
    )

    img_bin = crop_to_foreground(img_bin, pad=8)

    return img_bin.astype(bool)


def preprocess_signature_skeleton_from_binary(img_bin):
    img_bin = morphology.remove_small_objects(
        img_bin.astype(bool),
        min_size=MIN_OBJECT_SIZE
    )

    skel = morphology.skeletonize(img_bin)
    skel = prune_skeleton(skel, iterations=PRUNING_ITERATIONS)

    return skel.astype(bool)


# ============================================================
# AUGMENTATION
# ============================================================

def augment_signature_array(img):
    img_gray = rgb_to_gray_and_normalize(img)

    h, w = img_gray.shape[:2]

    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.90, 1.10)
    tx = random.uniform(-8.0, 8.0)
    ty = random.uniform(-8.0, 8.0)

    center_x = w / 2.0
    center_y = h / 2.0

    t_center_1 = transform.SimilarityTransform(
        translation=(-center_x, -center_y)
    )

    t_aug = transform.SimilarityTransform(
        scale=scale,
        rotation=np.deg2rad(angle),
        translation=(tx, ty)
    )

    t_center_2 = transform.SimilarityTransform(
        translation=(center_x, center_y)
    )

    tform = t_center_1 + t_aug + t_center_2

    img_aug = transform.warp(
        img_gray,
        inverse_map=tform.inverse,
        output_shape=(h, w),
        order=1,
        mode="constant",
        cval=1.0,
        preserve_range=True
    )

    return img_aug.astype(np.float32)


# ============================================================
# DESCRIPTEURS
# ============================================================

def skeleton_topology_features(skel, area):
    skel_len = int(np.sum(skel))

    if skel_len == 0:
        return [0, 0, 0.0, 0.0, 0.0, 0.0]

    neighbors = count_8_neighbors(skel)

    endpoints = int(np.sum(skel & (neighbors == 1)))
    branchpoints = int(np.sum(skel & (neighbors >= 3)))

    endpoint_ratio = safe_div(endpoints, skel_len)
    branchpoint_ratio = safe_div(branchpoints, skel_len)

    skel_norm_area = safe_div(skel_len, np.sqrt(area))
    skel_density = safe_div(skel_len, area)

    return [
        endpoints,
        branchpoints,
        endpoint_ratio,
        branchpoint_ratio,
        skel_norm_area,
        skel_density
    ]


def normalized_fourier_descriptors(binary_img):
    if binary_img is None or np.sum(binary_img) == 0:
        return np.zeros(N_FOURIER_DESCRIPTORS, dtype=np.float32)

    contours = measure.find_contours(binary_img.astype(float), 0.5)

    if len(contours) == 0:
        return np.zeros(N_FOURIER_DESCRIPTORS, dtype=np.float32)

    contour = max(contours, key=lambda c: c.shape[0])

    if contour.shape[0] < 5:
        return np.zeros(N_FOURIER_DESCRIPTORS, dtype=np.float32)

    pts = np.zeros((contour.shape[0], 2), dtype=np.float32)
    pts[:, 0] = contour[:, 1]
    pts[:, 1] = contour[:, 0]

    pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    dist = np.sqrt(np.sum(diffs ** 2, axis=1))
    curv_abs = np.concatenate([[0.0], np.cumsum(dist)])

    total_length = curv_abs[-1]

    if total_length <= 1e-8:
        return np.zeros(N_FOURIER_DESCRIPTORS, dtype=np.float32)

    samples = np.linspace(
        0,
        total_length,
        N_FOURIER_POINTS,
        endpoint=False
    )

    x = np.interp(samples, curv_abs, pts[:, 0])
    y = np.interp(samples, curv_abs, pts[:, 1])

    z = x + 1j * y

    z = z - np.mean(z)

    coeffs = np.fft.fft(z)

    desc = np.abs(coeffs[1:N_FOURIER_DESCRIPTORS + 1])

    denom = desc[0] if desc[0] > 1e-8 else np.max(desc)

    if denom <= 1e-8:
        return np.zeros(N_FOURIER_DESCRIPTORS, dtype=np.float32)

    desc = desc / denom

    return desc.astype(np.float32)


def extract_shape_features_from_binary(binary_img):
    if binary_img is None or binary_img.size == 0 or np.sum(binary_img) == 0:
        return np.zeros(46, dtype=np.float32)

    binary_img = binary_img.astype(bool)

    label_all = binary_img.astype(np.uint8)
    props_list = measure.regionprops(label_all)

    if len(props_list) == 0:
        return np.zeros(46, dtype=np.float32)

    p = props_list[0]

    area = float(p.area)

    if area <= 0:
        return np.zeros(46, dtype=np.float32)

    perimeter = float(measure.perimeter(binary_img, neighborhood=8))

    minr, minc, maxr, maxc = p.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    bbox_area = max(1.0, float(bbox_h * bbox_w))

    convex_area = float(p.convex_area) if p.convex_area > 0 else area

    major = float(p.major_axis_length)
    minor = float(p.minor_axis_length)

    eccentricity = float(p.eccentricity)
    solidity = safe_div(area, convex_area)
    extent = safe_div(area, bbox_area)

    compactness = safe_div(perimeter ** 2, area)
    circularity = safe_div(4.0 * np.pi * area, perimeter ** 2)

    bbox_ratio = safe_div(min(bbox_h, bbox_w), max(bbox_h, bbox_w))
    axis_ratio = safe_div(minor, major)

    euler_number = float(p.euler_number)

    connected_components = float(measure.label(binary_img, connectivity=2).max())
    ink_density = safe_div(area, binary_img.size)

    region_features = [
        eccentricity,
        solidity,
        extent,
        compactness,
        circularity,
        bbox_ratio,
        axis_ratio,
        euler_number,
        connected_components,
        ink_density
    ]

    skel = preprocess_signature_skeleton_from_binary(binary_img)
    skel_features = skeleton_topology_features(skel, area)

    nu = p.moments_normalized

    nu_indices = [
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 0),
        (0, 3),
        (2, 1),
        (1, 2)
    ]

    nu_features = []

    for i, j in nu_indices:
        if i < nu.shape[0] and j < nu.shape[1]:
            value = float(nu[i, j])
        else:
            value = 0.0

        if not np.isfinite(value):
            value = 0.0

        nu_features.append(value)

    try:
        hu = measure.moments_hu(nu)
        hu = np.array(hu, dtype=np.float64)
        hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        hu = np.nan_to_num(hu, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        hu = np.zeros(7, dtype=np.float64)

    fourier = normalized_fourier_descriptors(binary_img)

    features = np.array(
        region_features
        + skel_features
        + nu_features
        + hu.tolist()
        + fourier.tolist(),
        dtype=np.float32
    )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_signature_features_from_array(img):
    binary_img = preprocess_signature_binary_from_array(img)
    return extract_shape_features_from_binary(binary_img)


# ============================================================
# DATASET
# ============================================================

def list_dataset(source_path):
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Dossier introuvable : {source_path}")

    student_ids = [
        d for d in os.listdir(source_path)
        if os.path.isdir(os.path.join(source_path, d))
    ]

    student_ids.sort()

    if len(student_ids) == 0:
        raise ValueError(f"Aucun dossier étudiant trouvé dans : {source_path}")

    labels_to_id = {
        str(i): student_id
        for i, student_id in enumerate(student_ids)
    }

    id_to_label = {
        student_id: i
        for i, student_id in enumerate(student_ids)
    }

    image_paths = []
    labels = []

    for student_id in student_ids:
        student_dir = os.path.join(source_path, student_id)

        files = [
            f for f in os.listdir(student_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        files.sort()

        if len(files) < 2:
            print(f"Attention : peu d'images pour {student_id} : {len(files)}")

        for filename in files:
            image_paths.append(os.path.join(student_dir, filename))
            labels.append(id_to_label[student_id])

    return image_paths, np.array(labels, dtype=np.int32), labels_to_id


def split_train_test_per_student(image_paths, labels, train_ratio=TRAIN_RATIO):
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    for label in sorted(np.unique(labels)):
        idx = np.where(labels == label)[0].tolist()
        random.shuffle(idx)

        n_train = int(round(len(idx) * train_ratio))

        if len(idx) >= 2:
            n_train = min(max(1, n_train), len(idx) - 1)

        for i in idx[:n_train]:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])

        for i in idx[n_train:]:
            test_paths.append(image_paths[i])
            test_labels.append(labels[i])

    return (
        train_paths,
        np.array(train_labels, dtype=np.int32),
        test_paths,
        np.array(test_labels, dtype=np.int32)
    )


def build_feature_matrix(image_paths, labels, training=False):
    X = []
    y = []

    for path, label in zip(image_paths, labels):
        try:
            img = io.imread(path)
        except Exception as e:
            print("Image illisible :", path, e)
            continue

        feat = extract_signature_features_from_array(img)
        X.append(feat)
        y.append(label)

        if training:
            for _ in range(AUGMENT_PER_IMAGE):
                img_aug = augment_signature_array(img)
                feat_aug = extract_signature_features_from_array(img_aug)
                X.append(feat_aug)
                y.append(label)

    if len(X) == 0:
        raise ValueError("Aucune feature extraite.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ============================================================
# TRAIN SVM
# ============================================================

def train_signature_svm():
    set_seed()
    ensure_dir(OUTPUT_DIR)

    image_paths, labels, labels_to_id = list_dataset(SOURCE_PATH)

    train_paths, train_labels, test_paths, test_labels = split_train_test_per_student(
        image_paths,
        labels,
        train_ratio=TRAIN_RATIO
    )

    print("Nombre total d'images :", len(image_paths))
    print("Nombre de classes     :", len(labels_to_id))
    print("Images train          :", len(train_paths))
    print("Images test           :", len(test_paths))

    print("\nExtraction des features train...")
    X_train, y_train = build_feature_matrix(
        train_paths,
        train_labels,
        training=True
    )

    print("Extraction des features test...")
    X_test, y_test = build_feature_matrix(
        test_paths,
        test_labels,
        training=False
    )

    print("Shape X_train :", X_train.shape)
    print("Shape X_test  :", X_test.shape)

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED
        )
    )

    print("\nEntraînement du SVM...")
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy test :", acc)

    print("\nRapport classification :")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    package = {
        "model": svm_model,
        "labels_to_id": labels_to_id,
        "feature_size": int(X_train.shape[1]),
        "feature_pipeline": "shape_descriptors_svm",
        "source_path": SOURCE_PATH
    }

    joblib.dump(package, MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_to_id, f, indent=4, ensure_ascii=False)

    print("\nModèle sauvegardé :")
    print(MODEL_PATH)

    print("\nLabels sauvegardés :")
    print(LABELS_PATH)


# ============================================================
# MAIN
# ============================================================

def main():
    train_signature_svm()


if __name__ == "__main__":
    main()