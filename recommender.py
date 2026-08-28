"""
recommender.py

Content-based game recommender: embeds cover art with a pretrained
EfficientNetB0 and recommends similar games via cosine similarity.

Usage:
    embeddings, valid_indices = build_embeddings(img_dir="game_images", n_games=len(lm_df))
    similar = find_similar_games(2314, embeddings, valid_indices, top_k=5)
"""

import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = 224  # EfficientNet's recommended input size


def load_base_model():
    """Load EfficientNetB0 with ImageNet weights, pooled to a 1280-dim vector."""
    return EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")


def load_and_preprocess(path: str):
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)


def build_embeddings(img_dir: str, n_games: int, base_model=None):
    """Compute an EfficientNet embedding for every available game cover image.

    Returns:
        embeddings_norm: L2-normalized embedding matrix, shape (n_valid, 1280)
        valid_indices: list mapping embedding row -> original game index
    """
    if base_model is None:
        base_model = load_base_model()

    embeddings = []
    valid_indices = []

    for idx in range(n_games):
        path = f"{img_dir}/{idx}.jpg"
        if not os.path.exists(path):
            continue
        img = load_and_preprocess(path)
        emb = base_model.predict(img, verbose=0)[0]  # shape: (1280,)
        embeddings.append(emb)
        valid_indices.append(idx)

    embeddings = np.array(embeddings)
    embeddings_norm = normalize(embeddings)
    print("Embedding matrix shape:", embeddings.shape)
    return embeddings_norm, valid_indices


def find_similar_games(game_idx: int, embeddings_norm: np.ndarray, valid_indices: list, top_k: int = 5):
    """Return the top_k game indices most similar in cover-art embedding space."""
    if game_idx not in valid_indices:
        print("No image for this game.")
        return []

    emb_row = valid_indices.index(game_idx)
    target = embeddings_norm[emb_row].reshape(1, -1)
    sims = cosine_similarity(target, embeddings_norm)[0]

    # Sort by similarity (highest first), skip index 0 (the game itself)
    top_indices = sims.argsort()[::-1][1:top_k + 1]
    return [valid_indices[i] for i in top_indices]


def show_game(idx: int, img_dir: str):
    """Display a game's cover art (requires matplotlib)."""
    import matplotlib.pyplot as plt

    path = f"{img_dir}/{idx}.jpg"
    img = load_img(path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    from data_pipeline import build_dataset

    IMG_DIR = "game_images"

    # Was previously a hardcoded placeholder (N_GAMES = 5000). Using the
    # actual cleaned dataset size instead, so the embedding pass covers
    # every game that survived data_pipeline's cleaning/filtering, not an
    # arbitrary guessed count.
    lm_df, tag_df, genre_df, platform_df, status_df = build_dataset("video_games.csv")
    n_games = len(lm_df)

    embeddings_norm, valid_indices = build_embeddings(IMG_DIR, n_games)

    game_id = 2314
    similar_ids = find_similar_games(game_id, embeddings_norm, valid_indices, top_k=5)
    print(f"Games similar to {game_id}:", similar_ids)

    show_game(game_id, IMG_DIR)
    for sid in similar_ids:
        show_game(sid, IMG_DIR)
