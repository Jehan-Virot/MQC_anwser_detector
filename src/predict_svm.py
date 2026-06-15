# predict_signature_svm.py
# Programme uniquement pour prédire l'ID étudiant à partir d'une signature.
#
# Il charge :
#   signature_model_svm/svm_shape.joblib
#   signature_model_svm/labels.json
#
# Il utilise la même pipeline de features que train_signature_svm.py.
#
# Utilisation directe :
#   python predict_signature_svm.py chemin/vers/signature.png
#
# Utilisation dans ton main :
#   from predict_signature_svm import (
#       load_model,
#       load_reference_signatures,
#       crop_signature_from_page,
#       predict_signature_id
#   )

import os
import sys
import json
import joblib
import numpy as np

from skimage import io

# Très important :
# on réutilise exactement la même fonction que pendant l'entraînement.
from training_svm_model import extract_signature_features_from_array


# ============================================================
# PARAMÈTRES
# ============================================================

OUTPUT_DIR = "signature_model_svm"
MODEL_PATH = os.path.join(OUTPUT_DIR, "svm_shape.joblib")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.json")

SIGNATURE_DECISION_THRESHOLD = 0.15


# ============================================================
# CHARGEMENT MODÈLE / LABELS
# ============================================================

def load_model(model_path=MODEL_PATH):
    """
    Charge le modèle SVM sauvegardé par train_signature_svm.py.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            f"Lance d'abord : python train_signature_svm.py"
        )

    return joblib.load(model_path)


def load_reference_signatures(labels_path=LABELS_PATH):
    """
    Charge labels.json.
    Format attendu :
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


# ============================================================
# CROP SIGNATURE DEPUIS UNE PAGE
# ============================================================

def crop_signature_from_page(img_gray, signature_box):
    """
    Extrait la zone signature depuis une image de page complète.

    signature_box attendu :
        x1, y1, x2, y2
    """
    if signature_box is None:
        return None

    x1, y1, x2, y2 = signature_box

    h, w = img_gray.shape[:2]

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 5 or bh <= 5:
        return None

    pad_x = int(0.08 * bw)
    pad_y = int(0.08 * bh)

    crop = img_gray[
        y1 + pad_y:y2 - pad_y,
        x1 + pad_x:x2 - pad_x
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# PRÉDICTION
# ============================================================

def predict_signature_id(
    signature_model,
    labels_to_id,
    signature_crop,
    threshold=SIGNATURE_DECISION_THRESHOLD
):
    """
    Prédit l'identité d'une signature.

    Entrées :
        signature_model : modèle chargé avec load_model()
        labels_to_id    : labels chargés avec load_reference_signatures()
        signature_crop  : image numpy de la signature
        threshold       : seuil minimal de confiance

    Retourne :
        best_id, best_score, all_scores
    """
    if signature_crop is None:
        return None, 0.0, {}

    # Le fichier joblib contient normalement un dictionnaire :
    # {
    #     "model": svm_model,
    #     "labels_to_id": labels_to_id,
    #     ...
    # }
    if isinstance(signature_model, dict):
        clf = signature_model["model"]

        if "labels_to_id" in signature_model:
            labels_to_id = signature_model["labels_to_id"]
    else:
        clf = signature_model

    features = extract_signature_features_from_array(signature_crop)
    x = np.expand_dims(features, axis=0)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x)[0]
        classes = clf.classes_

        best_index = int(np.argmax(probs))
        best_label = int(classes[best_index])
        best_score = float(probs[best_index])

        best_id = labels_to_id[str(best_label)]

        all_scores = {}

        for cls, prob in zip(classes, probs):
            cls = int(cls)
            key = str(cls)

            if key in labels_to_id:
                all_scores[labels_to_id[key]] = float(prob)

        if best_score < threshold:
            return None, best_score, all_scores

        return best_id, best_score, all_scores

    pred_label = int(clf.predict(x)[0])
    best_id = labels_to_id[str(pred_label)]

    return best_id, 1.0, {best_id: 1.0}


def predict_signature_from_path(image_path, threshold=SIGNATURE_DECISION_THRESHOLD):
    """
    Prédit une signature directement depuis un fichier image.
    Utile pour tester rapidement le modèle.
    """
    signature_model = load_model()
    labels_to_id = load_reference_signatures()

    img = io.imread(image_path)

    return predict_signature_id(
        signature_model,
        labels_to_id,
        img,
        threshold=threshold
    )


# ============================================================
# MAIN TEST SIMPLE
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Utilisation :")
        print("  python predict_signature_svm.py chemin/vers/signature.png")
        return

    image_path = sys.argv[1]

    predicted_id, score, all_scores = predict_signature_from_path(
        image_path,
        threshold=0.0
    )

    print("Image :", image_path)
    print("ID signature prédit :", predicted_id)
    print("Score :", score)

    print("\nTop 5 scores :")
    top_scores = sorted(
        all_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]

    for student_id, student_score in top_scores:
        print(student_id, ":", student_score)


if __name__ == "__main__":
    main()