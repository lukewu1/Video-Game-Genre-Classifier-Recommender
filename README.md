# Video Game Genre Classifier & Recommender

An end-to-end ML pipeline that scrapes and cleans ~107,000 games from the
RAWG API (2005–2025), trains a multi-input neural network to classify game
genre from cover art + metadata, and powers a content-based recommender
using image embeddings.

## Overview

- **Data source:** RAWG API, ~107,000 games scraped across 2005–2025, with
  metadata, ratings, tags, platforms, and cover art.
- **Pipeline:** scrape → clean → feature engineer → correlation analysis →
  model training → evaluation → recommendation.

## What it does

1. **Data collection**
   - Scraped ~107,000 games from the RAWG API across chunked date ranges,
     with retry logic and incremental CSV merging (deduplicated by game id).

2. **Data cleaning & feature engineering**
   - Resolved missing values per-column based on data semantics (e.g.
     `stores`/`tags` filled with empty lists, `metacritic` dropped due to
     high null rate, `esrb_rating` filled as "Unknown" rather than dropped).
   - Ran correlation analysis to catch redundant/collinear features
     (`reviews_count` vs `ratings_count`, `rating` vs `rating_top`) before
     modeling.
   - Parsed nested JSON-like string columns (tags, genres, platforms) into
     structured lists, then one-hot encoded the top 30 most common tags.

3. **Genre classifier**
   - Multi-input Keras model: a CNN branch processes cover art
     (Conv2D → MaxPool → Dense), a separate dense branch processes tabular
     metadata, concatenated and passed through a final classification head.
   - Trained with an 80/20 stratified train/test split.

4. **Recommender**
   - Used a pretrained EfficientNetB0 (ImageNet weights) to generate
     1280-dim embeddings from game cover art.
   - Cosine similarity over normalized embeddings surfaces the top-k most
     visually/thematically similar games to a given title.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, EfficientNetB0,
RAWG API, Matplotlib/Seaborn

## Results

- Games scraped: ~107,000
- Test Accuracy: 0.1865


## Setup

```bash
export RAWG_API_KEY = your_api_key_here
pip install -r requirements.txt
python scraper.py       # optional -- only if you need to rebuild video_games.csv
python classifier.py
```

## Project structure
