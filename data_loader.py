"""
data_loader.py
==============

Dataset & data-collection module for the Sentiment Analysis Dashboard.

Supported data sources
----------------------
1. Amazon product reviews   - public datasets (e.g. Amazon Reviews 2023 by McAuley Lab,
                              hosted on Hugging Face) loaded from a local/remote CSV or JSONL.
2. Social media posts       - authorized APIs only (X/Twitter API v2 with a bearer token,
                              Reddit via the official API using PRAW credentials).
3. News articles/headlines  - NewsAPI.org (API key) or any public RSS feed (published for
                              syndication by the publisher).
4. User-uploaded CSV files  - via Streamlit's file uploader or a local path.
5. Sample demo dataset      - 500+ realistic generated records, no API keys required.

No unauthorized scraping is performed anywhere in this module. All real-world collectors
call official, documented endpoints and require the user's own credentials, which are read
from environment variables or Streamlit secrets - never hard-coded.
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_COLUMNS = ["ID", "Date", "Source", "Product", "Text", "Rating", "Sentiment", "Emotion"]

DATA_SOURCES = {
    "sample": "Sample demo dataset",
    "csv": "Upload CSV file",
    "amazon": "Amazon product reviews (public dataset)",
    "social": "Social media posts (authorized API)",
    "news": "News articles & headlines (NewsAPI / RSS)",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_dataset.csv"

# Keywords used to auto-detect column roles in arbitrary CSVs.
_ROLE_HINTS = {
    "text": ["text", "review", "content", "body", "comment", "message", "tweet", "post",
             "headline", "title", "description", "summary"],
    "date": ["date", "time", "timestamp", "created", "published", "posted"],
    "source": ["source", "platform", "channel", "origin", "site", "publisher"],
    "product": ["product", "item", "asin", "brand", "category", "topic", "subject"],
    "rating": ["rating", "score", "stars", "star", "overall"],
}


# ---------------------------------------------------------------------------
# Sample dataset generation
# ---------------------------------------------------------------------------

_PRODUCTS = {
    "Electronics": ["Echo Dot Smart Speaker", "Kindle Paperwhite", "Fire TV Stick 4K",
                    "Wireless Earbuds Pro", "Noise Cancelling Headphones", "Smartwatch Series 5",
                    "USB-C Fast Charger", "Portable Power Bank", "Bluetooth Speaker Mini",
                    "4K Action Camera"],
    "Home": ["Robot Vacuum Cleaner", "Air Fryer XL", "Espresso Machine", "Smart LED Bulbs",
             "Memory Foam Pillow", "Stainless Steel Cookware Set"],
    "Tech Brands": ["iPhone 16", "Galaxy S25", "Pixel 9", "Tesla Model 3", "PlayStation 5",
                    "MacBook Air M3"],
    "Topics": ["Stock Market", "Climate Policy", "AI Regulation", "Electric Vehicles",
               "Interest Rates", "Space Exploration"],
}

_TEMPLATES = {
    "Positive": {
        "Joy": [
            "Absolutely love the {p}! It exceeded every expectation I had.",
            "The {p} is fantastic - setup took two minutes and it just works.",
            "Best purchase this year. The {p} makes my day so much easier.",
            "So happy with the {p}, the quality is outstanding for the price.",
            "Can't stop smiling, the {p} arrived early and works perfectly.",
        ],
        "Trust": [
            "I've used the {p} for three months now and it is still rock solid.",
            "Reliable and well built, the {p} is a brand I'll keep buying from.",
            "Customer support for the {p} was quick and genuinely helpful.",
        ],
        "Surprise": [
            "Wow, did not expect the {p} to be this good at such a low price!",
            "Pleasantly surprised by how long the battery on the {p} lasts.",
        ],
    },
    "Negative": {
        "Anger": [
            "Terrible experience with the {p}. It broke after one week.",
            "The {p} is a complete waste of money, support ignored my emails.",
            "Furious - the {p} stopped working right after the return window closed.",
            "Worst product ever. The {p} overheats and the seller won't refund.",
        ],
        "Sadness": [
            "Really disappointed with the {p}, I had such high hopes for it.",
            "Sadly the {p} didn't live up to the reviews. Returning it.",
            "Unfortunately the {p} stopped charging. Such a letdown.",
        ],
        "Fear": [
            "The {p} started smoking while charging - honestly scary.",
            "Worried about privacy, the {p} keeps sending data without asking.",
        ],
        "Disgust": [
            "The {p} arrived dirty and used. Gross and unacceptable.",
            "Cheap plastic, awful smell - the {p} feels disgusting to hold.",
        ],
    },
    "Neutral": {
        "Neutral": [
            "The {p} is okay. Does what it says, nothing more.",
            "Average {p}. Some features are good, others are missing.",
            "Received the {p} today. Will update after more use.",
            "The {p} works as described. Packaging was standard.",
            "It's a decent {p}, though similar to others on the market.",
        ],
    },
}

_SOCIAL_TEMPLATES = {
    "Positive": {"Joy": ["Just got my {p} and I'm obsessed 😍 #happy",
                         "Shoutout to the team behind {p}, absolute game changer! 🙌"],
                 "Trust": ["Been using {p} daily, never let me down. Highly recommend 👍"]},
    "Negative": {"Anger": ["Seriously {p}?? Third update this week and it's still broken 😡",
                           "Never buying {p} again. Customer service is a joke."],
                 "Sadness": ["Kinda bummed, my {p} died today 😢"]},
    "Neutral": {"Neutral": ["Anyone else tried the new {p}? Thinking about it.",
                            "Saw an ad for {p} today. Thoughts?"]},
}

_NEWS_TEMPLATES = {
    "Positive": {"Joy": ["{p} sees record growth as investors cheer strong quarter",
                         "Breakthrough announced in {p}, experts hail major progress"],
                 "Trust": ["Analysts upgrade outlook on {p} citing stable fundamentals"]},
    "Negative": {"Fear": ["Concerns mount over {p} as uncertainty grips markets",
                          "Experts warn of risks in {p} amid growing volatility"],
                 "Anger": ["Public backlash grows over controversial {p} decision"],
                 "Sadness": ["{p} suffers steep decline, thousands of jobs at risk"]},
    "Neutral": {"Neutral": ["{p}: what to expect in the coming week",
                            "Officials to discuss {p} at upcoming summit",
                            "Report outlines latest developments in {p}"]},
}

_RATING_BY_SENTIMENT = {"Positive": [4, 5], "Neutral": [3], "Negative": [1, 2]}


def generate_sample_dataset(n_records: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic demo dataset with the standard columns.

    Mix: ~55% Amazon reviews, ~25% social posts, ~20% news headlines.
    A handful of intentional duplicates and missing values are injected so the
    missing-value and duplicate detectors have something to report.
    """
    rng = random.Random(seed)
    start = datetime(2025, 1, 1)
    rows = []

    for i in range(n_records):
        r = rng.random()
        if r < 0.55:
            source = "Amazon Reviews"
            product = rng.choice(_PRODUCTS["Electronics"] + _PRODUCTS["Home"])
            templates = _TEMPLATES
        elif r < 0.80:
            source = rng.choice(["Twitter/X", "Reddit"])
            product = rng.choice(_PRODUCTS["Tech Brands"] + _PRODUCTS["Electronics"][:4])
            templates = _SOCIAL_TEMPLATES
        else:
            source = rng.choice(["NewsAPI", "RSS Feed"])
            product = rng.choice(_PRODUCTS["Topics"] + _PRODUCTS["Tech Brands"][:3])
            templates = _NEWS_TEMPLATES

        sentiment = rng.choices(["Positive", "Negative", "Neutral"], weights=[0.48, 0.32, 0.20])[0]
        emotion = rng.choice(list(templates[sentiment].keys()))
        text = rng.choice(templates[sentiment][emotion]).format(p=product)

        rating = rng.choice(_RATING_BY_SENTIMENT[sentiment]) if source == "Amazon Reviews" else None
        date = start + timedelta(days=rng.randint(0, 364), hours=rng.randint(0, 23),
                                 minutes=rng.randint(0, 59))

        rows.append({
            "ID": f"REC-{i + 1:05d}",
            "Date": date.strftime("%Y-%m-%d %H:%M"),
            "Source": source,
            "Product": product,
            "Text": text,
            "Rating": rating,
            "Sentiment": sentiment,
            "Emotion": emotion,
        })

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)

    # Inject a few missing values (realistic data-quality issues).
    for idx in rng.sample(range(n_records), 8):
        df.at[idx, "Product"] = None
    for idx in rng.sample(range(n_records), 4):
        df.at[idx, "Text"] = None

    # Inject duplicates (same content, new IDs) + exact duplicates.
    dup_src = df.sample(6, random_state=seed).copy()
    dup_src["ID"] = [f"REC-{n_records + k + 1:05d}" for k in range(len(dup_src))]
    exact = df.sample(3, random_state=seed + 1)
    df = pd.concat([df, dup_src, exact], ignore_index=True)
    df["Rating"] = df["Rating"].astype("Int64")
    return df


def load_sample_dataset(regenerate: bool = False) -> pd.DataFrame:
    """Load the bundled demo dataset (creating it on first run)."""
    if regenerate or not SAMPLE_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_sample_dataset().to_csv(SAMPLE_PATH, index=False)
    return pd.read_csv(SAMPLE_PATH)


# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------

def load_csv(file, encoding: Optional[str] = None, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV from a path, bytes or a file-like object (e.g. Streamlit UploadedFile).

    Tries several encodings and auto-detects the delimiter when `sep` is None.
    """
    if isinstance(file, (str, Path)):
        raw = Path(file).read_bytes()
    elif isinstance(file, (bytes, bytearray)):
        raw = bytes(file)
    else:
        raw = file.getvalue() if hasattr(file, "getvalue") else file.read()

    if not raw:
        raise ValueError("The uploaded file is empty.")

    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python",
                             on_bad_lines="skip")
            if df.shape[1] == 0:
                raise ValueError("No columns found in file.")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.EmptyDataError as e:
            raise ValueError("The uploaded file has no data.") from e
    raise ValueError(f"Could not decode the file: {last_err}")


# ---------------------------------------------------------------------------
# Real-world collectors (authorized APIs / public datasets only)
# ---------------------------------------------------------------------------

def _get_secret(name: str) -> Optional[str]:
    """Read a credential from env vars or Streamlit secrets (never hard-coded)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        import streamlit as st  # noqa: WPS433
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None


def load_amazon_reviews(path_or_url: str, limit: int = 5000) -> pd.DataFrame:
    """Load Amazon reviews from a *public* dataset file (CSV, JSON or JSONL).

    Recommended sources (publicly released for research):
      - Amazon Reviews 2023 (McAuley Lab): https://amazon-reviews-2023.github.io/
        e.g. https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
      - Kaggle "Amazon Fine Food Reviews" / "Amazon Product Reviews" (download CSV).
    Columns are mapped to the standard schema where possible.
    """
    p = str(path_or_url)
    if p.endswith((".jsonl", ".json.gz", ".jsonl.gz")):
        df = pd.read_json(p, lines=True, nrows=limit)
    elif p.endswith(".json"):
        df = pd.read_json(p).head(limit)
    else:
        df = pd.read_csv(p, nrows=limit)

    mapping = {
        "text": "Text", "reviewText": "Text", "Text": "Text", "review_body": "Text",
        "rating": "Rating", "overall": "Rating", "Score": "Rating", "star_rating": "Rating",
        "parent_asin": "Product", "asin": "Product", "ProductId": "Product", "product_title": "Product",
        "timestamp": "Date", "reviewTime": "Date", "Time": "Date", "review_date": "Date",
    }
    out = pd.DataFrame()
    for src, dst in mapping.items():
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src]
    if "Date" in out.columns and pd.api.types.is_numeric_dtype(out["Date"]):
        unit = "ms" if out["Date"].max() > 1e11 else "s"
        out["Date"] = pd.to_datetime(out["Date"], unit=unit, errors="coerce")
    out["Source"] = "Amazon Reviews"
    return standardize(out)


def fetch_twitter_posts(query: str, max_results: int = 100) -> pd.DataFrame:
    """Fetch recent posts through the official X (Twitter) API v2.

    Requires env var / secret ``TWITTER_BEARER_TOKEN`` from developer.x.com.
    """
    import requests

    token = _get_secret("TWITTER_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Set TWITTER_BEARER_TOKEN to use the X/Twitter API.")
    resp = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": f"{query} -is:retweet lang:en",
                "max_results": max(10, min(max_results, 100)),
                "tweet.fields": "created_at"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame({
        "ID": [d["id"] for d in data],
        "Date": [d.get("created_at") for d in data],
        "Text": [d.get("text") for d in data],
    })
    df["Source"], df["Product"] = "Twitter/X", query
    return standardize(df)


def fetch_reddit_posts(query: str, subreddit: str = "all", limit: int = 100) -> pd.DataFrame:
    """Fetch posts via Reddit's official API (PRAW).

    Requires ``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET`` and ``REDDIT_USER_AGENT``.
    """
    import praw  # optional dependency

    cid, secret = _get_secret("REDDIT_CLIENT_ID"), _get_secret("REDDIT_CLIENT_SECRET")
    ua = _get_secret("REDDIT_USER_AGENT") or "sentiment-dashboard/1.0"
    if not (cid and secret):
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to use the Reddit API.")
    reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=ua)
    rows = [{
        "ID": s.id,
        "Date": datetime.utcfromtimestamp(s.created_utc),
        "Text": f"{s.title}. {s.selftext}".strip(),
        "Rating": None,
    } for s in reddit.subreddit(subreddit).search(query, limit=limit)]
    df = pd.DataFrame(rows)
    df["Source"], df["Product"] = "Reddit", query
    return standardize(df)


def fetch_news_newsapi(query: str, page_size: int = 100, language: str = "en") -> pd.DataFrame:
    """Fetch news headlines via NewsAPI.org. Requires ``NEWSAPI_KEY``."""
    import requests

    key = _get_secret("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("Set NEWSAPI_KEY to use NewsAPI.org.")
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": query, "pageSize": min(page_size, 100), "language": language,
                "sortBy": "publishedAt", "apiKey": key},
        timeout=20,
    )
    resp.raise_for_status()
    arts = resp.json().get("articles", [])
    df = pd.DataFrame({
        "Date": [a.get("publishedAt") for a in arts],
        "Text": [" - ".join(filter(None, [a.get("title"), a.get("description")])) for a in arts],
        "Publisher": [(a.get("source") or {}).get("name") for a in arts],
    })
    df["Source"], df["Product"] = "NewsAPI", query
    return standardize(df.drop(columns=["Publisher"]))


def fetch_news_rss(feed_url: str, topic: str = "News") -> pd.DataFrame:
    """Read a public RSS/Atom feed (published by the outlet for syndication).

    No key required. Example: https://feeds.bbci.co.uk/news/rss.xml
    """
    import feedparser  # optional dependency

    feed = feedparser.parse(feed_url)
    df = pd.DataFrame([{
        "Date": e.get("published") or e.get("updated"),
        "Text": " - ".join(filter(None, [e.get("title"), e.get("summary")])),
    } for e in feed.entries])
    df["Source"], df["Product"] = "RSS Feed", topic
    return standardize(df)


# ---------------------------------------------------------------------------
# Column selection / mapping
# ---------------------------------------------------------------------------

def detect_column(df: pd.DataFrame, role: str) -> Optional[str]:
    """Guess which column best fits a role: text, date, source, product, rating."""
    hints = _ROLE_HINTS.get(role, [])
    lower = {c: c.lower() for c in df.columns}
    for h in hints:  # exact match first
        for c, lc in lower.items():
            if lc == h:
                return c
    for h in hints:  # then substring
        for c, lc in lower.items():
            if h in lc:
                return c
    if role == "text":  # fallback: longest average string column
        obj = df.select_dtypes(include="object")
        if not obj.empty:
            return obj.apply(lambda s: s.astype(str).str.len().mean()).idxmax()
    if role == "rating":
        for c in df.select_dtypes(include="number").columns:
            s = df[c].dropna()
            if not s.empty and s.between(0, 10).all() and s.nunique() <= 11:
                return c
    return None


def suggest_mapping(df: pd.DataFrame) -> dict:
    return {role: detect_column(df, role) for role in _ROLE_HINTS}


@dataclass
class ColumnMapping:
    text: str
    date: Optional[str] = None
    source: Optional[str] = None
    product: Optional[str] = None
    rating: Optional[str] = None
    extra: list = field(default_factory=list)  # other columns to keep as-is


def apply_mapping(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Rename user-selected columns to the standard schema."""
    if mapping.text not in df.columns:
        raise ValueError(f"Text column '{mapping.text}' not found.")
    out = pd.DataFrame(index=df.index)
    out["Text"] = df[mapping.text]
    for role, std in [("date", "Date"), ("source", "Source"),
                      ("product", "Product"), ("rating", "Rating")]:
        col = getattr(mapping, role)
        if col and col in df.columns:
            out[std] = df[col]
    for c in ["ID", "Sentiment", "Emotion"]:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    for c in mapping.extra:
        if c in df.columns and c not in out.columns:
            out[c] = df[c]
    return standardize(out)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard columns exist with sensible types; keep extras at the end."""
    df = df.copy()
    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    if df["ID"].isna().all():
        df["ID"] = [f"REC-{i + 1:05d}" for i in range(len(df))]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False, format="mixed")
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Text"] = df["Text"].astype("string").str.strip()
    extras = [c for c in df.columns if c not in STANDARD_COLUMNS]
    return df[STANDARD_COLUMNS + extras].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset information & quality checks
# ---------------------------------------------------------------------------

def dataset_info(df: pd.DataFrame) -> dict:
    """High-level summary of the dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "text_columns": df.select_dtypes(include=["object", "string"]).columns.tolist(),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, missing %, unique values and an example."""
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Non-Null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique": df.nunique(dropna=True).values,
        "Example": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else ""
                    for c in df.columns],
    })


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that contain missing values (including blank strings)."""
    blanks = df.apply(lambda s: s.astype("string").str.strip().eq("").fillna(False).sum()
                      if s.dtype == object or str(s.dtype) == "string" else 0)
    rep = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Blank strings": blanks.values,
    })
    rep["Total"] = rep["Missing"] + rep["Blank strings"]
    rep["Percent"] = (rep["Total"] / max(len(df), 1) * 100).round(2)
    return rep[rep["Total"] > 0].sort_values("Total", ascending=False).reset_index(drop=True)


def detect_duplicates(df: pd.DataFrame, subset: Optional[Iterable[str]] = None,
                      keep: str | bool = "first") -> pd.DataFrame:
    """Return duplicate records. ``subset`` e.g. ["Text"] finds duplicate content."""
    subset = list(subset) if subset else None
    mask = df.duplicated(subset=subset, keep=keep)
    return df[mask]


def handle_missing(df: pd.DataFrame, strategy: str = "drop_text") -> pd.DataFrame:
    """Strategies: drop_text (drop rows with no Text), drop_any, fill_unknown."""
    if strategy == "drop_text":
        return df[df["Text"].notna() & df["Text"].astype("string").str.strip().ne("")].reset_index(drop=True)
    if strategy == "drop_any":
        return df.dropna().reset_index(drop=True)
    if strategy == "fill_unknown":
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object or str(out[c].dtype) == "string":
                out[c] = out[c].fillna("Unknown")
        return out
    raise ValueError(f"Unknown strategy: {strategy}")


def remove_duplicates(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=list(subset) if subset else None).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="data")
    return buf.getvalue()


def to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


if __name__ == "__main__":
    data = load_sample_dataset(regenerate=True)
    print(f"Sample dataset written to {SAMPLE_PATH} ({len(data)} rows)")
    print(dataset_info(data))
