import os
import json
import numpy as np
from tensorflow import keras
from skimage import io, color, transform, filters, morphology
from scipy import ndimage as ndi

IMG_HEIGHT = 239
IMG_WIDTH = 342
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

SIGNATURE_MODEL_PATH = "data/src/signature_model/model.keras"
SIGNATURE_DECISION_THRESHOLD = 0.50
PRUNING_ITERATIONS = 12

_LAST_MODEL_DIR = None


def rgb_to_gray_and_normalize(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def prune_skeleton(skel, n_iter=PRUNING_ITERATIONS):
    skel = skel.astype(bool)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    pruned = skel.copy()
    for _ in range(n_iter):
        nb_neighbors = ndi.convolve(pruned.astype(np.uint8), kernel, mode="constant", cval=0)
        endpoints = pruned & (nb_neighbors <= 1)
        if not np.any(endpoints):
            break
        pruned[endpoints] = False

    return pruned

def resize(img_gray, target_h=IMG_HEIGHT, target_w=IMG_WIDTH):
    img_resized = transform.resize(
        img_gray,
        (target_h, target_w),
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)

    return img_resized

def preprocess_signature_for_model_from_array(img):
    img_gray = rgb_to_gray_and_normalize(img)

    if img_gray.shape != (IMG_HEIGHT, IMG_WIDTH):
            img_gray = resize(img_gray)

    img_smooth = filters.gaussian(img_gray, sigma=1.0, preserve_range=True)
    threshold = filters.threshold_otsu(img_smooth)

    img_bin = img_smooth < threshold
    img_bin = morphology.remove_small_objects(img_bin, max_size=8)

    img_skel = morphology.skeletonize(img_bin)
    img_pruned = prune_skeleton(img_skel)

    return np.expand_dims(img_pruned.astype(np.float32), axis=-1)


def preprocess_signature_for_model_from_path(path):
    return preprocess_signature_for_model_from_array(io.imread(path))


def crop_signature_from_page(img_gray, signature_box):
    if signature_box is None:
        return None

    x1, y1, x2, y2 = signature_box
    h, w = img_gray.shape[:2]

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 5 or bh <= 5:
        return None

    pad_x = int(0.08 * bw)
    pad_y = int(0.08 * bh)

    crop = img_gray[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

    if crop.size == 0:
        return None

    return crop


# On garde le même nom pour éviter de modifier ton main.
def load_model(model_path=SIGNATURE_MODEL_PATH):
    global _LAST_MODEL_DIR
    _LAST_MODEL_DIR = os.path.dirname(model_path)
    return keras.models.load_model(model_path, compile=False)


# On garde le même nom pour éviter de modifier ton main.
# Pour le CNN softmax, cette fonction charge juste labels.json.
def load_reference_signatures(_unused_path=None):
    if _LAST_MODEL_DIR is None:
        labels_path = "data/src/signature_model/labels.json"
    else:
        labels_path = os.path.join(_LAST_MODEL_DIR, "labels.json")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_to_id = json.load(f)

    return labels_to_id


def predict_signature_id(signature_model, labels_to_id, signature_crop, threshold=SIGNATURE_DECISION_THRESHOLD):
    """
    CNN softmax :
    signature inconnue -> probabilité par classe -> ID le plus probable.
    Retourne :
    - student_id prédit
    - score softmax
    - dictionnaire {student_id: score}
    """
    if signature_crop is None:
        return None, 0.0, {}

    img = preprocess_signature_for_model_from_array(signature_crop)
    x = np.expand_dims(img, axis=0)

    probs = signature_model.predict(x, verbose=0)[0]

    best_label = int(np.argmax(probs))
    best_score = float(probs[best_label])

    best_id = labels_to_id[str(best_label)]

    all_scores = {
        labels_to_id[str(i)]: float(probs[i])
        for i in range(len(probs))
        if str(i) in labels_to_id
    }

    if best_score < threshold:
        return None, best_score, all_scores

    return best_id, best_score, all_scores