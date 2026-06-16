import numpy as np
import joblib
from os import path
from functions.SVM_pipeline_functions import extract_signature_features_from_array, crop_trim_signature


MODEL_DIR = "signature_model_svm"
MODEL_PATH = path.join(MODEL_DIR, "svm_shape.joblib")
LABELS_PATH = path.join(MODEL_DIR, "labels.json")
THRESHOLD_FACTOR = 25 #important as the models are perfect images, while the ones in the photos are still imperfect and very different even after pre processing)

def load_similarity_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def select_features_from_model(features, model):
    return features[model["feature_keep"]]


def similarity_from_distance(distance, threshold):
    """
    Score lisible entre 0 et 1.
    1 = très proche.
    0 = loin.
    """
    if not np.isfinite(distance):
        return 0.0

    return float(np.exp(- (distance / max(threshold, 1e-8)) ** 2))


def verify_signature_id(model, student_id_grid, signature_crop):
    """
    Vérifie si la signature correspond à l'ID lu dans la grille.

    Retour :
    - student_id_grid si accepté
    - None si rejeté
    - score entre 0 et 1
    - infos debug
    """
    if signature_crop is None or student_id_grid is None:
        return None, 0.0, {}

    student_id_grid = str(student_id_grid)

    gallery = model["gallery"]
    scaler = model["scaler"]
    threshold = model["threshold"]

    if student_id_grid not in gallery:
        return None, 0.0, {
            "reason": "student_id_absent_from_gallery",
            "student_id_grid": student_id_grid
        }

    features = extract_signature_features_from_array(signature_crop)
    features = select_features_from_model(features, model)

    x = scaler.transform(np.expand_dims(features, axis=0))[0]

    ref_vectors = gallery[student_id_grid]

    dists = np.mean(np.abs(ref_vectors - x), axis=1)

    k = min(3, len(dists))
    nearest = np.partition(dists, k - 1)[:k]
    distance = float(np.mean(nearest))

    score = similarity_from_distance(distance, threshold)

    accepted = distance <= threshold * THRESHOLD_FACTOR

    debug = {
        "student_id_grid": student_id_grid,
        "distance": distance,
        "threshold": float(threshold),
        "score": score,
        "accepted": accepted
    }

    if not accepted:
        return None, score, debug

    return student_id_grid, score, debug


def predict_signature_from_box(img_gray, signature_box, model, student_id_grid):
    if signature_box is None:
        return None, 0.0

    try:
        signature_crop = crop_trim_signature(img_gray, signature_box)

        if signature_crop is None:
            return None, 0.0

        student_id_signature, score, debug = verify_signature_id(
            model=model,
            student_id_grid=student_id_grid,
            signature_crop=signature_crop
        )

        return student_id_signature, score

    except Exception as e:
        print("Erreur signature similarity :", e)
        return None, 0.0