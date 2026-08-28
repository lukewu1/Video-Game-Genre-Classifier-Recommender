"""
data_pipeline.py

Loads the raw RAWG video game dataset, cleans it, and engineers features
(tags, genres, platforms, added_by_status) used by classifier.py and
recommender.py.

Usage:
    from data_pipeline import load_and_clean_data, engineer_features
    lm_df, tag_df, genre_df, platform_df, status_df = build_dataset("video_games.csv")
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load the raw RAWG CSV and drop/fill columns based on data quality analysis.

    Column decisions (see project README for full rationale):
      - saturated_color / dominant_color: single unique value, no signal -> drop
      - tba, score, clip, user_game, short_screenshots, parent_platforms -> drop
      - platforms / stores / added_by_status / tags: fill missing with "[]"
      - metacritic: too many nulls -> drop
      - esrb_rating: fill missing as "Unknown" (often indicates indie/PC titles)
      - community_rating: only contains NaN/0, no signal -> drop
    """
    data = pd.read_csv(csv_path)

    data = data.drop(columns=[
        'saturated_color',
        'dominant_color',
        'tba',
        'score',
        'clip',
        'user_game',
        'short_screenshots',
        'parent_platforms',
    ], errors="ignore")

    data["platforms"] = data["platforms"].fillna("[]")
    data["stores"] = data["stores"].fillna("[]")
    data["added_by_status"] = data["added_by_status"].fillna("[]")
    data = data.drop(columns=["metacritic"], errors="ignore")
    data["tags"] = data["tags"].fillna("[]")
    data["esrb_rating"] = data["esrb_rating"].fillna("Unknown")
    data = data.drop(columns=['community_rating'], errors="ignore")

    return data


# ---------------------------------------------------------------------------
# Parsing helpers for stringified list/dict columns
# ---------------------------------------------------------------------------

def extract_slugs(x):
    """Parse a stringified list of dicts (e.g. tags/genres) and pull out 'slug'."""
    try:
        parsed = ast.literal_eval(x)
        return [d['slug'] for d in parsed]
    except Exception:
        return []


def extract_platform_slugs(x):
    """Parse the platforms column, which nests slug under a 'platform' key."""
    try:
        parsed = ast.literal_eval(x)
        return [d['platform']['slug'] for d in parsed if 'platform' in d]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(data: pd.DataFrame, top_n_tags: int = 30):
    """Build the modeling dataframe: parse list columns, one-hot encode
    tags/genres/platforms, scale added_by_status, and log-transform skewed
    count features.

    Returns:
        lm_df: cleaned + feature-engineered dataframe (rating > 0 only)
        tag_df, genre_df, platform_df, status_df: one-hot / scaled feature frames
    """
    lm_df = data.copy()

    # RAWG sets rating to 0 when a game has no ratings yet -> drop those rows
    lm_df = lm_df[lm_df['rating'] > 0]

    # Drop any stale one-hot columns from a previous run
    cols_to_drop = [c for c in lm_df.columns
                     if c.startswith('tag_') or c.startswith('genre_') or c.startswith('platforms_')]
    lm_df = lm_df.drop(columns=cols_to_drop, errors='ignore')

    # added_by_status: parse dict string -> columns, scale
    lm_df['added_by_status_parsed'] = lm_df['added_by_status'].apply(
        lambda x: ast.literal_eval(x)
    )
    status_df = lm_df['added_by_status_parsed'].apply(pd.Series).fillna(0.0)
    scaler = StandardScaler()
    scaled_status = scaler.fit_transform(status_df)
    status_df = pd.DataFrame(scaled_status, columns=status_df.columns, index=lm_df.index)

    # Parse tags / genres / platforms into slug lists
    lm_df['tag_list'] = lm_df['tags'].apply(extract_slugs)
    lm_df['genre_list'] = lm_df['genres'].apply(extract_slugs)
    lm_df['platforms_list'] = lm_df['platforms'].apply(extract_platform_slugs)

    # Keep only the N most common tags (long tail is too sparse to be useful)
    tag_counts = Counter(slug for tags in lm_df['tag_list'] for slug in tags)
    top_tags = set(slug for slug, _ in tag_counts.most_common(top_n_tags))
    lm_df['tag_filtered'] = lm_df['tag_list'].apply(
        lambda tags: [t for t in tags if t in top_tags]
    )

    # TODO: the original notebook also referenced a `genre_filtered` column
    # (used later when building the primary_genre label) but the cell that
    # created it was missing from the exported script. Recreate it here,
    # analogous to tag_filtered, using all genres (genre lists are short
    # enough that no top-N filtering is needed) -- adjust if you filtered
    # genres differently in the original notebook.
    lm_df['genre_filtered'] = lm_df['genre_list']

    # One-hot encode tags / genres / platforms
    mlb_tags = MultiLabelBinarizer()
    tag_matrix = mlb_tags.fit_transform(lm_df['tag_filtered'])
    tag_df = pd.DataFrame(
        tag_matrix, columns=[f"tag_{t}" for t in mlb_tags.classes_], index=lm_df.index
    )

    mlb_genres = MultiLabelBinarizer()
    genre_matrix = mlb_genres.fit_transform(lm_df['genre_list'])
    genre_df = pd.DataFrame(
        genre_matrix, columns=[f"genre_{t}" for t in mlb_genres.classes_], index=lm_df.index
    )

    mlb_platforms = MultiLabelBinarizer()
    platform_matrix = mlb_platforms.fit_transform(lm_df['platforms_list'])
    platform_df = pd.DataFrame(
        platform_matrix, columns=[f"platform_{t}" for t in mlb_platforms.classes_], index=lm_df.index
    )

    lm_df = pd.concat([lm_df, tag_df, genre_df, platform_df, status_df], axis=1)

    # Log-transform heavily skewed count features (huge min/max spread)
    skewed_cols = ['suggestions_count', 'added', 'reviews_count', 'reviews_text_count']
    for col in skewed_cols:
        lm_df[col + "_log"] = np.log1p(lm_df[col])

    # TODO: the original notebook also used 'ratings_count_log' and
    # 'playtime_log' later (in the image-classifier metadata features) but
    # never logged them in the visible cells. Adding them here for
    # consistency -- confirm against your original notebook if the log
    # transform should apply to these too.
    lm_df['ratings_count_log'] = np.log1p(lm_df['ratings_count'])
    lm_df['playtime_log'] = np.log1p(lm_df['playtime'])

    return lm_df, tag_df, genre_df, platform_df, status_df


def build_dataset(csv_path: str, top_n_tags: int = 30):
    """Convenience wrapper: load, clean, and engineer features in one call."""
    data = load_and_clean_data(csv_path)
    return engineer_features(data, top_n_tags=top_n_tags)


if __name__ == "__main__":
    lm_df, tag_df, genre_df, platform_df, status_df = build_dataset("video_games.csv")
    print("Rows after cleaning:", len(lm_df))
    print("Tag features:", tag_df.shape[1])
    print("Genre features:", genre_df.shape[1])
    print("Platform features:", platform_df.shape[1])
