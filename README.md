# Sentiment Analysis Dashboard – Data Module

```
pip install -r requirements.txt
python data_loader.py        # (re)generates data/sample_dataset.csv
streamlit run app.py
```

## Files
- `data_loader.py` – all loading, column mapping, quality checks and export logic (Pandas).
- `app.py` – Streamlit app; the **Data** page uses the module, the cleaned dataset is shared via `st.session_state["df"]`.
- `data/sample_dataset.csv` – 609 demo records (ID, Date, Source, Product, Text, Rating, Sentiment, Emotion) with a few intentional missing values and duplicates.

## Integrating into an existing Streamlit app
```python
import data_loader as dl
df = dl.load_sample_dataset()            # or dl.load_csv(uploaded_file) + dl.apply_mapping(...)
st.session_state["df"] = df
```
You can also call `data_page()` from `app.py` inside your own navigation.

## Real-world sources (authorized only – no scraping)
| Source | Function | Credential |
|---|---|---|
| Amazon reviews (public datasets, e.g. Amazon Reviews 2023) | `load_amazon_reviews(path_or_url)` | none |
| X / Twitter API v2 | `fetch_twitter_posts(query)` | `TWITTER_BEARER_TOKEN` |
| Reddit official API | `fetch_reddit_posts(query)` | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |
| NewsAPI.org | `fetch_news_newsapi(query)` | `NEWSAPI_KEY` |
| Public RSS feeds | `fetch_news_rss(url)` | none |

Set credentials as environment variables or in `.streamlit/secrets.toml`.
