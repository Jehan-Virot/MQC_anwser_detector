from skimage import filters, morphology, transform, measure, io
import os
import joblib
import json
import numpy as np
import random
from functions.some_morphology_functions import count_8_neighbors
from functions.outils_generaux_images import rgb_to_gray_and_normalize, stretch_histo, safe_div

#parameters
PRUNING_ITERATIONS = 8
MIN_OBJECT_SIZE = 8
AUGMENT_PER_IMAGE = 4
TRAIN_RATIO = 0.80

####################################################################################################
#sauvegarde et load des modèles

def load_model(model_path="../signature_model_svm/svm_shape.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Lance d'abord : python train_signature_svm.py"
        )

    return joblib.load(model_path)


def load_reference_signatures(labels_path="../signature_model_svm/labels.json"):
    """
    format du json : 
        {
            "0": "19283",
            "1": "50305",
            ...
        }
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"labels.json introuvable : {labels_path}\n"
            f"Lance d'abord : python train_signature_svm.py"
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_to_id = json.load(f)

    return labels_to_id


######################################################################################################
#function of the pipeline 
#crop l'image à l'endroit de la boite
#trim le crop pour retirer les bordures noires
#filtre gaussien
#apply histrogram stretching and otsu dans cette zone
#skeleton et prunning (pour garder la forme generale de la signature + pour extraire features endpoints et branches)


#crop et trimming
def crop_trim_signature(img_gray, signature_box, trim_ratio=0.07):
    if signature_box is None:
        return None

    x1, y1, x2, y2 = signature_box

    h_img, w_img = img_gray.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)

    crop = img_gray[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    trim_y = int(trim_ratio * h)
    trim_x = int(trim_ratio * w)

    crop = crop[trim_y:h-trim_y, trim_x:w-trim_x]

    if crop.size == 0:
        return None

    return crop


#skeleton et pruning
def prune_skeleton(skeleton, iterations=PRUNING_ITERATIONS):
    pruned = skeleton.astype(bool).copy()

    for _ in range(iterations):
        neighbors = count_8_neighbors(pruned)
        endpoints = pruned & (neighbors == 1)

        if not np.any(endpoints):
            break

        pruned[endpoints] = False

    return pruned

def preprocess_signature_skeleton_from_binary(img_bin):
    img_bin = morphology.remove_small_objects(img_bin.astype(bool),max_size=MIN_OBJECT_SIZE-1)
    skel = morphology.skeletonize(img_bin)
    skel = prune_skeleton(skel, iterations=PRUNING_ITERATIONS)

    return skel.astype(bool)

#
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

#preprocessing de la signature
def preprocess_signature_binary_from_array(img):
    img_gray = rgb_to_gray_and_normalize(img)
    img_gray = stretch_histo(img_gray)

    img_smooth = filters.gaussian(img_gray,sigma=1.0,preserve_range=True)

    try:
        threshold = filters.threshold_otsu(img_smooth)
    except ValueError:
        return np.zeros_like(img_smooth, dtype=bool)

    img_bin = img_smooth < threshold
    img_bin = morphology.remove_small_objects(img_bin.astype(bool),max_size=MIN_OBJECT_SIZE-1)
    img_bin = morphology.closing(img_bin,np.ones((2, 2), dtype=bool))
    img_bin = crop_to_foreground(img_bin, pad=8)

    return img_bin.astype(bool)


########################################################
#data augmentation


def augment_signature_array(img):
    img_gray = rgb_to_gray_and_normalize(img)

    h, w = img_gray.shape[:2]

    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.90, 1.10)
    tx = random.uniform(-8.0, 8.0)
    ty = random.uniform(-8.0, 8.0)

    center_x = w / 2.0
    center_y = h / 2.0

    t_center_1 = transform.SimilarityTransform(translation=(-center_x, -center_y))
    t_aug = transform.SimilarityTransform(scale=scale,rotation=np.deg2rad(angle),translation=(tx, ty))
    t_center_2 = transform.SimilarityTransform(translation=(center_x, center_y))
    tform = t_center_1 + t_aug + t_center_2

    img_aug = transform.warp(img_gray, inverse_map=tform.inverse, output_shape=(h, w), order=1, mode="constant", cval=1.0,preserve_range=True)

    return img_aug.astype(np.float32)


########################################################
#descriptors
#on garde les descriptors non sensibles au scaling, rotation et translation


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
        return np.zeros(16, dtype=np.float32)

    contours = measure.find_contours(binary_img.astype(float), 0.5)

    if len(contours) == 0:
        return np.zeros(16, dtype=np.float32)

    contour = max(contours, key=lambda c: c.shape[0])

    if contour.shape[0] < 5:
        return np.zeros(16, dtype=np.float32)

    pts = np.zeros((contour.shape[0], 2), dtype=np.float32)
    pts[:, 0] = contour[:, 1]
    pts[:, 1] = contour[:, 0]

    pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    dist = np.sqrt(np.sum(diffs ** 2, axis=1))
    curv_abs = np.concatenate([[0.0], np.cumsum(dist)])

    total_length = curv_abs[-1]

    if total_length <= 1e-8:
        return np.zeros(16, dtype=np.float32)

    samples = np.linspace(0, total_length, 128, endpoint=False)

    x = np.interp(samples, curv_abs, pts[:, 0])
    y = np.interp(samples, curv_abs, pts[:, 1])

    z = x + 1j * y

    z = z - np.mean(z)

    coeffs = np.fft.fft(z)

    desc = np.abs(coeffs[1:16 + 1])

    denom = desc[0] if desc[0] > 1e-8 else np.max(desc)

    if denom <= 1e-8:
        return np.zeros(16, dtype=np.float32)

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

    convex_area = float(p.area_convex) if p.area_convex > 0 else area

    major = float(p.axis_major_length)
    minor = float(p.axis_minor_length)

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


##############################################################################
#dataset

def list_dataset(source_path):
    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Dossier introuvable : {source_path}")

    student_ids = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    student_ids.sort()

    if len(student_ids) == 0:
        raise ValueError(f"Aucun dossier étudiant trouvé dans : {source_path}")

    labels_to_id = {str(i): student_id for i, student_id in enumerate(student_ids)}
    id_to_label = {student_id: i for i, student_id in enumerate(student_ids)}

    image_paths = []
    labels = []

    for student_id in student_ids:
        student_dir = os.path.join(source_path, student_id)

        files = [f for f in os.listdir(student_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
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
