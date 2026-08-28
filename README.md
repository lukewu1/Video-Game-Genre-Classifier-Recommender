# Video Game Genre Classifier & Recommender

An end-to-end ML pipeline that cleans and engineers features from a 25K+ game
dataset (RAWG), trains a multi-input neural network to classify game genre
from cover art + metadata, and powers a content-based recommender using
image embeddings.

## Overview

- **Data source:** RAWG video game dataset (2005–2025), ~[N] games with
  metadata, ratings, tags, platforms, and cover art.
- **Pipeline:** cleaning → feature engineering → correlation analysis →
  model training → evaluation → recommendation.

## What it does

1. **Data cleaning & feature engineering**
   - Resolved missing values per-column based on data semantics (e.g.
     `stores`/`tags` filled with empty lists, `metacritic` dropped due to
     high null rate, `esrb_rating` filled as "Unknown" rather than dropped).
   - Ran correlation analysis to catch redundant/collinear features
     (`reviews_count` vs `ratings_count`, `rating` vs `rating_top`) before
     modeling.
   - Parsed nested JSON-like string columns (tags, genres, platforms) into
     structured lists, then one-hot encoded the top 30 most common tags.

2. **Genre classifier**
   - Multi-input Keras model: a CNN branch processes cover art
     (Conv2D → MaxPool → Dense), a separate dense branch processes tabular
     metadata, concatenated and passed through a final classification head.
   - Trained with an 80/20 stratified train/test split.

3. **Recommender**
   - Used a pretrained EfficientNetB0 (ImageNet weights) to generate
     1280-dim embeddings from game cover art.
   - Cosine similarity over normalized embeddings surfaces the top-k most
     visually/thematically similar games to a given title.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, EfficientNetB0,
Matplotlib/Seaborn

## Results

- Test accuracy: 0.1865

<img width="803" height="715" alt="Screenshot 2026-08-27 200805" src="https://github.com/user-attachments/assets/06c3f1a8-7515-4007-9030-3e2a9e858000" />

## Setup

```bash
pip install -r requirements.txt
python classifier.py
```

## Project structure
├── data_pipeline.py # cleaning, feature engineering
├── classifier.py # multi-input CNN + metadata model
├── recommender.py # EfficientNet embeddings + cosine similarity
├── requirements.txt
└── README.md

## Notes

The components in this repo (data pipeline, classifier, recommender) reflect my individual
contribution.
