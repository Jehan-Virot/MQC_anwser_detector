import os
import json
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.preprocessing import StandardScaler
from functions.SVM_pipeline_functions import *
from functions.miscellaneous import *
from functions.outils_generaux_images import set_seed

SOURCE_PATH = "data/STUDENT_CLASS_SIGNATURES"
OUTPUT_DIR = "signature_model_similarity"
MODEL_PATH = os.path.join(OUTPUT_DIR, "signature_similarity.joblib")

FEATURE_KEEP = (
    [0, 1, 2, 3, 4, 6, 7, 8]
    + [10, 11, 12, 13, 14]
    + list(range(23, 30))
    + list(range(30, 46))
)


def select_features(features):
    return features[FEATURE_KEEP]


def distance_to_gallery(x, gallery_vectors, k=3):
    """
    Distance robuste : moyenne des k plus proches voisins.
    Plus la distance est petite, plus la signature est similaire.
    """
    if gallery_vectors is None or len(gallery_vectors) == 0:
        return np.inf

    dists = np.linalg.norm(gallery_vectors - x, axis=1)

    k = min(k, len(dists))
    nearest = np.partition(dists, k - 1)[:k]

    return float(np.mean(nearest))


def calibrate_global_threshold(X_scaled, student_ids):
    """
    Calibre un seuil global à partir :
    - des distances positives : même étudiant
    - des distances négatives : étudiant différent

    Acceptation si distance <= threshold.
    """
    student_ids = np.array(student_ids)
    positive_distances = []
    negative_distances = []

    unique_ids = sorted(set(student_ids))

    for i, x in enumerate(X_scaled):
        sid = student_ids[i]

        same_mask = student_ids == sid
        diff_mask = student_ids != sid

        same_vectors = X_scaled[same_mask]
        diff_vectors = X_scaled[diff_mask]

        # distance positive en leave-one-out
        same_indices = np.where(same_mask)[0]
        local_index = np.where(same_indices == i)[0][0]

        if len(same_vectors) > 1:
            same_dists = np.linalg.norm(same_vectors - x, axis=1)
            same_dists[local_index] = np.inf
            positive_distances.append(float(np.min(same_dists)))

        # distance négative : plus proche imposteur
        if len(diff_vectors) > 0:
            negative_distances.append(distance_to_gallery(x, diff_vectors, k=1))

    positive_distances = np.array(positive_distances, dtype=np.float32)
    negative_distances = np.array(negative_distances, dtype=np.float32)

    # seuil simple et robuste :
    # on accepte environ 95% des vrais matchs, tout en évitant les imposteurs proches
    pos_thr = float(np.percentile(positive_distances, 95))
    neg_thr = float(np.percentile(negative_distances, 5))

    threshold = 0.5 * (pos_thr + neg_thr)

    stats = {
        "positive_mean": float(np.mean(positive_distances)),
        "positive_p95": pos_thr,
        "negative_mean": float(np.mean(negative_distances)),
        "negative_p05": neg_thr,
        "threshold": float(threshold)
    }

    return float(threshold), stats


def build_similarity_model():
    set_seed()
    ensure_dir(OUTPUT_DIR)

    image_paths, labels, labels_to_id = list_dataset(SOURCE_PATH)

    X_raw = []
    student_ids = []

    for path, label in zip(image_paths, labels):
        try:
            img = io.imread(path)
        except Exception as e:
            print("Image illisible :", path, e)
            continue

        student_id = labels_to_id[str(int(label))]

        # image originale
        feat = extract_signature_features_from_array(img)
        X_raw.append(select_features(feat))
        student_ids.append(student_id)

        # augmentations légères : rotation, scale, translation
        for _ in range(AUGMENT_PER_IMAGE):
            img_aug = augment_signature_array(img)
            feat_aug = extract_signature_features_from_array(img_aug)
            X_raw.append(select_features(feat_aug))
            student_ids.append(student_id)

    X_raw = np.array(X_raw, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    threshold, stats = calibrate_global_threshold(X_scaled, student_ids)

    gallery = {}

    for x, sid in zip(X_scaled, student_ids):
        if sid not in gallery:
            gallery[sid] = []
        gallery[sid].append(x)

    for sid in gallery:
        gallery[sid] = np.array(gallery[sid], dtype=np.float32)

    package = {
        "type": "signature_similarity_model",
        "gallery": gallery,
        "scaler": scaler,
        "threshold": threshold,
        "stats": stats,
        "feature_keep": FEATURE_KEEP,
        "source_path": SOURCE_PATH
    }

    joblib.dump(package, MODEL_PATH)

    with open(os.path.join(OUTPUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print("Modèle de similarité sauvegardé :", MODEL_PATH)
    print("Stats calibration :")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    build_similarity_model()


