# train_signature_cnn_skeleton_softmax.py
# Entraîne un CNN softmax pour reconnaître l'ID étudiant à partir d'une signature.
#
# Architecture attendue :
# data/STUDENT_CLASS_SIGNATURES/{id_student}/{id_student}_{xxx}.png
#
# Sorties :
# signature_model/model.keras   : modèle CNN softmax
# signature_model/labels.json   : {"0": "19283", ...}
#
# Prétraitement conservé :
# image -> gris -> gaussien -> Otsu -> binaire -> skeleton -> pruning
# convention : fond = 0, encre = 1

import os
import json
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import skimage as ski
import keras
from keras import layers, models, callbacks


# ============================================================
# PARAMÈTRES
# ============================================================

SOURCE_PATH = "data/STUDENT_CLASS_SIGNATURES"
OUTPUT_DIR = "signature_model"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.keras")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.json")

# Format réel des signatures : largeur 342, hauteur 239.
IMG_HEIGHT = 239
IMG_WIDTH = 342
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

TRAIN_RATIO = 0.80

RANDOM_SEED = 4564

EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.0007
AUGMENT_PER_IMAGE = 5
BATCH_SIZE = 32

PRUNE_ITERATIONS = 8
MIN_OBJECT_SIZE = 8
BATCH_SIZE = 32


# ============================================================
# OUTILS IMAGE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def rgb_to_gray_and_normalize(img):
    if img.ndim == 3:
        img = ski.color.rgb2gray(img)

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def count_8_neighbors(binary_img):
    """Compte les voisins 8-connexes pour chaque pixel foreground."""
    padded = np.pad(binary_img.astype(np.uint8), 1, mode="constant")
    neighbors = np.zeros(binary_img.shape, dtype=np.uint8)

    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            neighbors += padded[
                dr:dr + binary_img.shape[0],
                dc:dc + binary_img.shape[1]
            ]

    return neighbors


def prune_skeleton(skeleton, iterations=8):
    """
    Pruning simple : supprime itérativement les extrémités du skeleton.
    Une extrémité = pixel du squelette avec exactement 1 voisin en 8-connexité.
    """
    pruned = skeleton.astype(bool).copy()

    for _ in range(iterations):
        neighbors = count_8_neighbors(pruned)
        endpoints = pruned & (neighbors == 1)

        if not np.any(endpoints):
            break

        pruned[endpoints] = False

    return pruned


def skeleton_pruning(binary_img, prune_iterations=PRUNE_ITERATIONS):
    """
    Binaire -> skeleton -> pruning.
    Objectif : garder surtout la forme globale de la signature.
    """
    binary_img = binary_img.astype(bool)
    binary_img = ski.morphology.remove_small_objects(
        binary_img,
        min_size=MIN_OBJECT_SIZE
    )

    skeleton = ski.morphology.skeletonize(binary_img)
    skeleton = prune_skeleton(skeleton, iterations=prune_iterations)

    return skeleton.astype(np.float32)


def preprocess_signature_image_from_array(img):
    """
    Prétraitement conservé :
    grayscale -> gaussian -> Otsu -> binaire -> skeleton + pruning.
    Convention CNN : fond = 0, encre = 1.
    """
    img_gray = rgb_to_gray_and_normalize(img)

    # Sécurité : toutes les signatures doivent finir en 239 x 342.
    if img_gray.shape != (IMG_HEIGHT, IMG_WIDTH):
        img_gray = ski.transform.resize(
            img_gray,
            (IMG_HEIGHT, IMG_WIDTH),
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

    img_smooth = ski.filters.gaussian(
        img_gray,
        sigma=1.0,
        preserve_range=True
    )

    threshold = ski.filters.threshold_otsu(img_smooth)
    img_bin = img_smooth < threshold

    img_shape = skeleton_pruning(img_bin)

    return np.expand_dims(img_shape.astype(np.float32), axis=-1)


def load_signature_image(path):
    img = ski.io.imread(path)
    return preprocess_signature_image_from_array(img)


def augment_signature(img):
    """
    Augmentation géométrique légère : rotation, translation, zoom.
    Comme l'image est skeletonisée, on rebinarise puis on reskeletonise légèrement.
    """
    img2d = img[:, :, 0]

    angle = random.uniform(-5.0, 5.0)
    scale = random.uniform(0.93, 1.07)
    tx = random.uniform(-7.0, 7.0)
    ty = random.uniform(-7.0, 7.0)

    h, w = img2d.shape
    center_x = w / 2.0
    center_y = h / 2.0

    t_center_1 = ski.transform.SimilarityTransform(
        translation=(-center_x, -center_y)
    )
    t_aug = ski.transform.SimilarityTransform(
        scale=scale,
        rotation=np.deg2rad(angle),
        translation=(tx, ty)
    )
    t_center_2 = ski.transform.SimilarityTransform(
        translation=(center_x, center_y)
    )

    tform = t_center_1 + t_aug + t_center_2

    img_aug = ski.transform.warp(
        img2d,
        inverse_map=tform.inverse,
        output_shape=(h, w),
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True
    )

    img_aug = img_aug > 0.25
    img_aug = skeleton_pruning(
        img_aug,
        prune_iterations=max(2, PRUNE_ITERATIONS // 2)
    )

    return np.expand_dims(img_aug.astype(np.float32), axis=-1)


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

    labels_to_id = {str(i): student_id for i, student_id in enumerate(student_ids)}
    id_to_label = {student_id: i for i, student_id in enumerate(student_ids)}

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
            print(f"Attention : très peu d'images pour {student_id} : {len(files)}")

        for filename in files:
            image_paths.append(os.path.join(student_dir, filename))
            labels.append(id_to_label[student_id])

    return image_paths, np.array(labels, dtype=np.int32), labels_to_id


def split_train_test_per_student(image_paths, labels, train_ratio=0.80):
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

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


class SignatureImageCache:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        if path not in self.cache:
            self.cache[path] = load_signature_image(path)
        return self.cache[path]


class BalancedSoftmaxSequence(keras.utils.Sequence):
    """
    Générateur pour CNN softmax.
    Chaque batch est équilibré autant que possible entre les classes.
    Avec AUGMENT_PER_IMAGE=20, on augmente virtuellement la longueur d'une époque.
    """

    def __init__(self, image_paths, labels, num_classes, training=True, batch_size=BATCH_SIZE):
        super().__init__()
        self.image_paths = list(image_paths)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = num_classes
        self.training = training
        self.batch_size = batch_size
        self.cache = SignatureImageCache()

        self.class_to_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in range(num_classes)
        }
        self.available_classes = [
            c for c in range(num_classes)
            if len(self.class_to_indices[c]) > 0
        ]

        multiplier = AUGMENT_PER_IMAGE if training else 1
        self.samples_per_epoch = len(self.image_paths) * multiplier
        self.n_batches = int(np.ceil(self.samples_per_epoch / self.batch_size))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, batch_index):
        xs = []
        ys = []

        for _ in range(self.batch_size):
            c = random.choice(self.available_classes)
            img_index = random.choice(self.class_to_indices[c])

            img = self.cache.get(self.image_paths[img_index])
            if self.training:
                img = augment_signature(img)

            xs.append(img)
            ys.append(c)

        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)


# ============================================================
# MODÈLE CNN SOFTMAX
# ============================================================

def build_signature_cnn(num_classes):
    """
    CNN classique : une image de signature -> classe étudiant.
    Sortie softmax : probabilité pour chaque ID connu.
    """
    model = models.Sequential(name="signature_cnn_skeleton_softmax")

    model.add(layers.Input(shape=INPUT_SHAPE))

    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Plus stable qu'un Flatten énorme sur 342 x 239.
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.40))

    model.add(layers.Dense(num_classes, activation="softmax", name="student_id_softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# INFÉRENCE UTILITAIRE
# ============================================================

def predict_student_id_from_signature_array(model, labels_to_id, img_array, top_k=3):
    """
    img_array : image déjà chargée, crop signature brut ou image signature.
    Retourne l'ID prédit, le score, et le top-k.
    """
    x = preprocess_signature_image_from_array(img_array)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]

    top = [
        (labels_to_id[str(int(i))], float(probs[i]))
        for i in top_indices
    ]

    best_id, best_score = top[0]
    return best_id, best_score, top


def predict_student_id_from_signature_path(model, labels_to_id, path, top_k=3):
    img = ski.io.imread(path)
    return predict_student_id_from_signature_array(model, labels_to_id, img, top_k=top_k)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths, labels, labels_to_id = list_dataset(SOURCE_PATH)
    num_classes = len(labels_to_id)

    print("Nombre d'étudiants :", num_classes)
    print("Nombre total d'images :", len(image_paths))
    print("Format image attendu :", INPUT_SHAPE)
    print("Augmentations virtuelles par image train :", AUGMENT_PER_IMAGE)

    train_paths, train_labels, test_paths, test_labels = split_train_test_per_student(
        image_paths,
        labels,
        TRAIN_RATIO
    )

    print("Images train :", len(train_paths))
    print("Images test  :", len(test_paths))

    train_seq = BalancedSoftmaxSequence(
        train_paths,
        train_labels,
        num_classes=num_classes,
        training=True,
        batch_size=BATCH_SIZE
    )

    test_seq = BalancedSoftmaxSequence(
        test_paths,
        test_labels,
        num_classes=num_classes,
        training=False,
        batch_size=BATCH_SIZE
    )

    model = build_signature_cnn(num_classes)
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00005
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_seq,
        validation_data=test_seq,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1
    )

    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_to_id, f, indent=4, ensure_ascii=False)

    test_loss, test_acc = model.evaluate(test_seq, verbose=0)

    print("\nModèle sauvegardé :")
    print(MODEL_PATH)
    print(LABELS_PATH)
    print(f"Accuracy test finale : {test_acc:.4f}")
    print(f"Loss test finale     : {test_loss:.4f}")


if __name__ == "__main__":
    main()
