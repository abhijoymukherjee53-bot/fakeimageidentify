"""
train.py — Trains a CNN to classify CIFAKE images as REAL or FAKE (AI-generated).

Expects the CIFAKE dataset (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
laid out as:

    data/
      train/
        REAL/*.jpg
        FAKE/*.jpg
      test/
        REAL/*.jpg
        FAKE/*.jpg

Run:
    python train.py
Produces:
    model/model.h5              — trained Keras model
    model/training_history.png  — accuracy/loss curves
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (32, 32)          # CIFAKE images are 32x32
BATCH_SIZE = 64
EPOCHS = 15
DATA_DIR = "data"
MODEL_DIR = "model"
SEED = 42


def build_datasets():
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Couldn't find '{train_dir}'. Download the CIFAKE dataset and arrange it as "
            "data/train/{REAL,FAKE} and data/test/{REAL,FAKE}. See README.md."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names  # ['FAKE', 'REAL'] alphabetical
    print(f"Classes: {class_names}  (0 -> {class_names[0]}, 1 -> {class_names[1]})")

    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=False,
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model():
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ], name="augmentation")

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Rescaling(1.0 / 255),
        data_augmentation,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="cifake_cnn")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = build_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "model.h5"), save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    if test_ds is not None:
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    # class_names[1] should be "REAL" since folders are alphabetical (FAKE, REAL)
    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        f.write(",".join(class_names))

    print(f"Model saved to {os.path.join(MODEL_DIR, 'model.h5')}")


if __name__ == "__main__":
    main()
