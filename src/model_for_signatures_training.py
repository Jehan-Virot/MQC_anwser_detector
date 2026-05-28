import tensorflow as tf
from tensorflow.keras import models, layers
import os
import json
import numpy as np
import skimage as ski
import random
from functions.load_image_functions import load_binary_image


SOURCE_PATH = "data/STUDENT_CLASS_SIGNATURES/"



def prepare_label_path(path):
    dir_par_id_eleve = os.listdir(path)
    dir_par_id_eleve.sort()
    
    labels_to_id = {i:id for i, id in enumerate(dir_par_id_eleve)}
    id_to_label = {id:i for i, id in enumerate(dir_par_id_eleve)}
    
    image_path = []
    labels = []
    
    for dir_eleve in dir_par_id_eleve:
        dir_eleve_path = path + dir_eleve
        list_signature_eleve = os.listdir(dir_eleve_path)
        
        for signature_eleve in list_signature_eleve:
            signature_eleve = dir_eleve_path + "/" + signature_eleve
            image_path.append(signature_eleve)
            labels.append(id_to_label[dir_eleve])
            
    return image_path, labels, labels_to_id

def train_validation_sep(images_paths, labels):
    random.seed(42)
    data_by_class = {}
    for path, label in zip(images_paths, labels):
        data_by_class.setdefault(label, []).append(path)

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for label, paths in data_by_class.items():
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * 0.2))
        val_p = paths[:n_val]
        train_p = paths[n_val:]
        for p in train_p:
            train_paths.append(p)
            train_labels.append(label)
        for p in val_p:
            val_paths.append(p)
            val_labels.append(label)

    return train_paths, train_labels, val_paths, val_labels

            
def cnn_model_pour_signature(nb_eleve):
    model = models.Sequential([
        layers.Input(shape=(342, 239, 1)),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        # Tu peux supprimer Dropout si tu veux rester ultra strict cours.
        layers.Dropout(0.3),

        layers.Dense(nb_eleve, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def load_images(paths, labels):
    X = []
    y = []

    for path, label in zip(paths, labels):
        try:
            img = load_binary_image(path, is_not_signature=False)
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"[WARNING] Image ignorée : {path} | {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y

def train_model(epochs=40, batch_size=16):
    image_paths, labels, label_to_id = prepare_label_path(SOURCE_PATH)

    train_paths, train_labels, val_paths, val_labels = train_validation_sep(image_paths,labels)

    print(f"Images totales : {len(image_paths)}")
    print(f"Train : {len(train_paths)}")
    print(f"Validation : {len(val_paths)}")
    print(f"Nombre d'élèves/classes : {len(label_to_id)}")

    X_train, y_train = load_images(train_paths, train_labels)
    X_val, y_val = load_images(val_paths, val_labels)

    num_classes = len(label_to_id)

    model = cnn_model_pour_signature(num_classes)

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join("/workspaces/MQC_anwser_detector//src/signature_model/", "signature_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    model_path = "/workspaces/MQC_anwser_detector/src/signature_model/model.keras"
    labels_path = "/workspaces/MQC_anwser_detector/src/signature_model/labels.json"
    model.save(model_path)

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=4)

    print("\nModèle sauvegardé :")
    print(model_path)
    print(labels_path)

    return model, history

train_model()