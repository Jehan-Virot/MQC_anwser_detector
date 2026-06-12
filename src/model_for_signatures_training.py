# train_signature_cnn.py
# Entraîne un CNN Keras pour reconnaître les signatures des étudiants.
#
# Architecture attendue :
# data/STUDENT_CLASS_SIGNATURES/{id_student}/{id_student}_{xxx}.png
#
# Sorties :
# signature_model/model.keras
# signature_model/labels.json
#
# labels.json est au format demandé :
# {
#   "0": "19283",
#   "1": "50305",
#   ...
# }
#
# Prétraitement :
# image -> gris -> filtre gaussien -> Otsu -> binaire
# convention : fond = 0, encre = 1
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os
import json
import random
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

# Tes images de signatures font 342 x 239.
# En numpy / Keras : (hauteur, largeur, canaux)
IMG_HEIGHT = 239
IMG_WIDTH = 342
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Split équilibré par élève : 20 images -> 16 train, 4 test.
TRAIN_RATIO = 0.80

# Augmentation légère : on ne veut pas transformer une signature en une autre.
AUGMENT_PER_IMAGE = 1

RANDOM_SEED = 42
EPOCHS = 60
LEARNING_RATE = 0.001
PATIENCE = 10


# ============================================================
# OUTILS IMAGE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def rgb_to_gray_and_normalize(img):
    """
    Convertit une image RGB ou grayscale en float [0, 1].
    """
    if img.ndim == 3:
        img = ski.color.rgb2gray(img)

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def preprocess_signature_image_from_array(img):
    """
    Prétraitement demandé :
    - grayscale
    - filtre gaussien
    - Otsu
    - binaire

    Convention CNN :
    fond = 0
    encre = 1
    """
    img_gray = rgb_to_gray_and_normalize(img)

    # Sécurité si une image n'a pas exactement la bonne taille.
    if img_gray.shape != (IMG_HEIGHT, IMG_WIDTH):
        img_gray = ski.transform.resize(
            img_gray,
            (IMG_HEIGHT, IMG_WIDTH),
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

    # Filtre gaussien vu en traitement spatial.
    img_smooth = ski.filters.gaussian(img_gray, sigma=1.0, preserve_range=True)

    # Otsu : pixels sombres = encre.
    threshold = ski.filters.threshold_otsu(img_smooth)
    img_bin = img_smooth < threshold

    # Nettoyage très léger pour éviter quelques pixels isolés.
    img_bin = ski.morphology.remove_small_objects(img_bin, min_size=8)

    # Keras attend float32.
    img_bin = img_bin.astype(np.float32)

    # Ajout du canal.
    img_bin = np.expand_dims(img_bin, axis=-1)

    return img_bin


def load_signature_image(path):
    img = ski.io.imread(path)
    return preprocess_signature_image_from_array(img)


def augment_signature(img):
    """
    Augmentation légère.
    img est au format (H, W, 1), fond = 0, encre = 1.

    On utilise uniquement des transformations géométriques simples :
    rotation, translation, zoom.
    """
    img2d = img[:, :, 0]

    angle = random.uniform(-5.0, 5.0)
    scale = random.uniform(0.93, 1.07)
    tx = random.uniform(-6.0, 6.0)
    ty = random.uniform(-6.0, 6.0)

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

    # On rebinarise après interpolation.
    img_aug = img_aug > 0.35
    img_aug = img_aug.astype(np.float32)

    return np.expand_dims(img_aug, axis=-1)


# ============================================================
# CHARGEMENT DATASET
# ============================================================

def list_dataset(source_path):
    """
    Retourne :
    - image_paths
    - y
    - labels_to_id : {"0": "id_student", ...}
    """
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
    """
    Split équilibré par classe.
    Avec 20 images par élève et train_ratio=0.80 :
    16 train / 4 test pour chaque élève.
    """
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    unique_labels = sorted(np.unique(labels))

    for label in unique_labels:
        idx = np.where(labels == label)[0].tolist()
        random.shuffle(idx)

        n_train = int(round(len(idx) * train_ratio))

        # Sécurité : on garde au moins 1 image de test si possible.
        if len(idx) >= 2:
            n_train = min(max(1, n_train), len(idx) - 1)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        for i in train_idx:
            train_paths.append(image_paths[i])
            train_labels.append(labels[i])

        for i in test_idx:
            test_paths.append(image_paths[i])
            test_labels.append(labels[i])

    return (
        train_paths,
        np.array(train_labels, dtype=np.int32),
        test_paths,
        np.array(test_labels, dtype=np.int32)
    )


class BalancedSignatureSequence(keras.utils.Sequence):
    """
    Générateur Keras équilibré.
    Objectif :
    - éviter des batchs déséquilibrés ;
    - chaque batch prend une image par étudiant ;
    - donc chaque batch contient toutes les classes.
    """

    def __init__(self, image_paths, labels, num_classes, training=True):
        super().__init__()

        self.image_paths = list(image_paths)
        self.labels = np.array(labels, dtype=np.int32)
        self.num_classes = num_classes
        self.training = training

        self.class_to_indices = {}

        for c in range(num_classes):
            self.class_to_indices[c] = np.where(self.labels == c)[0].tolist()

        self.batch_size = num_classes

        max_images_per_class = max(len(v) for v in self.class_to_indices.values())

        # IMPORTANT :
        # ne pas appeler ça self.num_batches,
        # car Keras utilise déjà ce nom en interne.
        self.n_batches = max_images_per_class

        self.on_epoch_end()

    def __len__(self):
        return self.n_batches

    def on_epoch_end(self):
        for c in range(self.num_classes):
            random.shuffle(self.class_to_indices[c])

    def __getitem__(self, batch_index):
        xs = []
        ys = []

        for c in range(self.num_classes):
            indices = self.class_to_indices[c]

            if len(indices) == 0:
                continue

            img_index = indices[batch_index % len(indices)]
            img = load_signature_image(self.image_paths[img_index])

            if self.training and AUGMENT_PER_IMAGE > 0:
                if random.random() < 0.60:
                    img = augment_signature(img)

            xs.append(img)
            ys.append(c)

        x_batch = np.array(xs, dtype=np.float32)
        y_batch = np.array(ys, dtype=np.int32)

        order = np.arange(len(y_batch))
        np.random.shuffle(order)

        return x_batch[order], y_batch[order]


# ============================================================
# MODÈLE CNN
# ============================================================

def build_signature_cnn(num_classes):
    """
    CNN simple et adapté à un petit dataset.

    Anti-overfitting :
    - modèle pas trop gros ;
    - MaxPooling pour réduire les cartes ;
    - Dropout ;
    - EarlyStopping dans l'entraînement ;
    - augmentation légère dans le générateur.
    """
    model = models.Sequential(name="signature_cnn")

    model.add(layers.Input(shape=INPUT_SHAPE))

    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.40))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


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

    train_paths, train_labels, test_paths, test_labels = split_train_test_per_student(
        image_paths,
        labels,
        TRAIN_RATIO
    )

    print("Images train :", len(train_paths))
    print("Images test  :", len(test_paths))

    train_seq = BalancedSignatureSequence(
        train_paths,
        train_labels,
        num_classes=num_classes,
        training=True
    )

    test_seq = BalancedSignatureSequence(
        test_paths,
        test_labels,
        num_classes=num_classes,
        training=False
    )

    model = build_signature_cnn(num_classes)
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00005
        ),
        callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
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

    # Sauvegarde finale du meilleur modèle restauré.
    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_to_id, f, indent=4, ensure_ascii=False)

    print("\nModèle sauvegardé :")
    print(MODEL_PATH)
    print(LABELS_PATH)

    test_loss, test_acc = model.evaluate(test_seq, verbose=0)
    print(f"Accuracy test finale : {test_acc:.4f}")
    print(f"Loss test finale     : {test_loss:.4f}")


if __name__ == "__main__":
    main()