"""
classifier.py

Trains a multi-input neural network that predicts a game's primary genre
from its cover art (CNN branch) and tabular metadata (dense branch).

Usage:
    from data_pipeline import build_dataset
    lm_df, tag_df, genre_df, platform_df, status_df = build_dataset("video_games.csv")
    model, history, label_encoder = train_classifier(lm_df, tag_df, img_dir="game_images")
"""

import os

import numpy as np
import pandas as pd
import PIL.Image as Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Input, layers, models

IMG_SIZE = 128


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_classifier_data(lm_df: pd.DataFrame, tag_df: pd.DataFrame, img_dir: str):
    """Filter to rows with an available cover image and a valid primary genre,
    then assemble the metadata feature matrix and encoded genre labels.
    """
    df = lm_df.copy()
    tags = tag_df.copy()

    df["img_index"] = df.index

    available_images = {
        int(f.split(".")[0])
        for f in os.listdir(img_dir)
        if f.endswith(".jpg") and f.split(".")[0].isdigit()
    }
    mask_has_image = df["img_index"].isin(available_images)
    df = df[mask_has_image]
    tags = tags[mask_has_image]

    # Primary genre = first genre in the filtered genre list
    df["primary_genre"] = df["genre_filtered"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    mask_has_genre = df["primary_genre"].notna()
    df = df[mask_has_genre]
    tags = tags[mask_has_genre]

    # Drop genres that appear fewer than 2 times (too rare to stratify/split on)
    counts = df["primary_genre"].value_counts()
    valid_genres = counts[counts >= 2].index
    df = df[df["primary_genre"].isin(valid_genres)]
    tags = tags.loc[df.index]

    df = df.reset_index(drop=True)
    tags = tags.reset_index(drop=True)

    meta_numeric = df[[
        'suggestions_count_log',
        'added_log',
        'ratings_count_log',
        'reviews_count_log',
        'reviews_text_count_log',
        'playtime_log',
    ]].astype("float32").reset_index(drop=True)

    tags_clean = tags.astype("float32").reset_index(drop=True)
    meta_features = pd.concat([meta_numeric, tags_clean], axis=1)
    X_meta = meta_features.to_numpy().astype("float32")

    y_str = df["primary_genre"].astype(str).reset_index(drop=True)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)

    return df, X_meta, y_encoded, label_encoder


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def load_image_safe(path):
    """Load an image for the tf.data pipeline; returns a black image if missing."""

    def _load_np(p):
        p = p.decode("utf-8")
        if not os.path.exists(p):
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        img = Image.open(p).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    img = tf.numpy_function(_load_np, [path], tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def make_dataset(image_paths, meta_data, labels, batch_size=32, shuffle=True):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(load_image_safe, num_parallel_calls=tf.data.AUTOTUNE)
    meta_ds = tf.data.Dataset.from_tensor_slices(meta_data)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip(((img_ds, meta_ds), label_ds))
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(meta_input_dim: int, num_classes: int) -> tf.keras.Model:
    """Multi-input model: CNN branch over cover art + dense branch over
    tabular metadata, concatenated into a final classification head.
    """
    # Image branch
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation="relu")(img_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    img_feat = layers.Dense(128, activation="relu")(x)
    img_feat = layers.Dropout(0.4)(img_feat)

    # Metadata branch
    meta_input = Input(shape=(meta_input_dim,))
    m = layers.Dense(128, activation="relu")(meta_input)
    m = layers.Dropout(0.3)(m)
    m = layers.Dense(64, activation="relu")(m)
    m = layers.Dropout(0.3)(m)

    # Combine
    combined = layers.concatenate([img_feat, m])
    z = layers.Dense(128, activation="relu")(combined)
    out = layers.Dense(num_classes, activation="softmax")(z)

    model = models.Model(inputs=[img_input, meta_input], outputs=out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_classifier(lm_df: pd.DataFrame, tag_df: pd.DataFrame, img_dir: str,
                      epochs: int = 10, batch_size: int = 32, test_size: float = 0.2):
    df, X_meta, y_encoded, label_encoder = prepare_classifier_data(lm_df, tag_df, img_dir)
    num_classes = len(label_encoder.classes_)

    print("Samples:", len(df))
    print("Meta features:", X_meta.shape)
    print("Labels:", y_encoded.shape, "classes:", num_classes)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=y_encoded
    )

    X_meta_train, X_meta_test = X_meta[train_idx], X_meta[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    train_image_paths = [f"{img_dir}/{df.loc[i, 'img_index']}.jpg" for i in train_idx]
    test_image_paths = [f"{img_dir}/{df.loc[i, 'img_index']}.jpg" for i in test_idx]

    missing = [p for p in train_image_paths + test_image_paths if not os.path.exists(p)]
    print("Missing image paths after cleaning:", len(missing))

    train_ds = make_dataset(train_image_paths, X_meta_train, y_train, batch_size=batch_size)
    test_ds = make_dataset(test_image_paths, X_meta_test, y_test, batch_size=batch_size, shuffle=False)

    model = build_model(meta_input_dim=X_meta.shape[1], num_classes=num_classes)
    model.summary()

    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

    # Evaluation: classification report on the held-out test set
    from sklearn.metrics import classification_report
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model, history, label_encoder


if __name__ == "__main__":
    from data_pipeline import build_dataset

    lm_df, tag_df, genre_df, platform_df, status_df = build_dataset("video_games.csv")
    model, history, label_encoder = train_classifier(lm_df, tag_df, img_dir="game_images")
    model.save("genre_classifier.keras")
