"""
scraper.py

Pulls game metadata from the RAWG API across date-bucketed ranges (2005-2025)
and writes/updates a local CSV, deduplicating by RAWG game id.

Requires a RAWG API key: https://rawg.io/apidocs
Set it as an environment variable before running:
    export RAWG_API_KEY=your_key_here

Usage:
    python scraper.py
"""

import os
import time

import pandas as pd
import requests

API_KEY = os.environ.get("RAWG_API_KEY")

# Date buckets to page through. RAWG paginates within a date range, so
# splitting the full window into chunks avoids hitting page-depth limits
# and keeps individual requests fast.
YEARS = [
    ("2005-01-01", "2007-01-01"),
    ("2007-01-01", "2009-01-01"),
    ("2009-01-01", "2011-01-01"),
    ("2011-01-01", "2013-01-01"),
    ("2013-01-01", "2015-01-01"),
    ("2015-01-01", "2017-01-01"),
    ("2017-01-01", "2019-01-01"),
    ("2019-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2025-12-31"),
]

PAGE_SIZE = 40
MAX_PAGES_PER_RANGE = 250
REQUEST_DELAY_SEC = 0.3
RETRY_DELAY_SEC = 0.5
MAX_RETRIES = 5


def fetch_rawg_page(start: str, end: str, page: int):
    """Fetch a single page of games released in [start, end).

    Returns:
        list of game dicts, [] if this range has no more pages,
        or None if the request failed after all retries.
    """
    url = "https://api.rawg.io/api/games"
    params = {
        "key": API_KEY,
        "dates": f"{start},{end}",
        "page_size": PAGE_SIZE,
        "page": page,
        "ordering": "-released",
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 404:
                return []  # no more pages in this range

            if r.status_code != 200:
                print(f"[{start} -> {end}] Page {page}: HTTP {r.status_code} "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY_SEC)
                continue

            try:
                data = r.json()
            except ValueError:
                print(f"[{start} -> {end}] Page {page}: JSON decode error "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY_SEC)
                continue

            return data.get("results", [])

        except requests.RequestException as e:
            print(f"[{start} -> {end}] Page {page}: Request error {e} "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY_SEC)

    print(f"[{start} -> {end}] Page {page}: FAILED after {MAX_RETRIES} attempts, skipping")
    return None


def scrape_all(years=YEARS) -> pd.DataFrame:
    """Page through every date bucket and collect all game records."""
    new_rows = []

    for start, end in years:
        print(f"Fetching games from {start} to {end}")
        for page in range(1, MAX_PAGES_PER_RANGE + 1):
            time.sleep(REQUEST_DELAY_SEC)
            data = fetch_rawg_page(start, end, page)
            if not data:
                print(f"Finished at page {page - 1}")
                break
            new_rows.extend(data)

    df_new = pd.DataFrame(new_rows)
    print("\nNew rows fetched:", len(df_new))
    return df_new


def save_or_merge(df_new: pd.DataFrame, csv_path: str):
    """Merge newly scraped rows into an existing CSV, deduplicating by id."""
    df_existing = None

    if os.path.exists(csv_path):
        try:
            df_existing = pd.read_csv(csv_path)
            print("Loaded existing CSV:", len(df_existing))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            print("CSV unreadable or empty -- creating new file.")

    if df_existing is not None:
        df_total = pd.concat([df_existing, df_new], ignore_index=True)
        df_total.drop_duplicates(subset=["id"], inplace=True)
        df_total.to_csv(csv_path, index=False)
        print("Updated CSV saved:", csv_path, "-- total rows:", len(df_total))
    else:
        df_new.to_csv(csv_path, index=False)
        print("Created new CSV:", csv_path, "-- total rows:", len(df_new))


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("Set the RAWG_API_KEY environment variable before running.")

    csv_path = "video_games.csv"
    df_new = scrape_all()
    save_or_merge(df_new, csv_path)
