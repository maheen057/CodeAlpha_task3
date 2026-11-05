"""
Sentiment Analysis Dashboard - Streamlit entry point.

Run:  streamlit run app.py

The "Data" page is powered by data_loader.py. The loaded, standardized dataset is
stored in st.session_state["df"] so the other dashboard pages (analysis, charts,
model) can read it with: df = st.session_state.get("df").
"""

import pandas as pd
import streamlit as st

import data_loader as dl

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _sample() -> pd.DataFrame:
    return dl.load_sample_dataset()


@st.cache_data(show_spinner=False)
def _csv(raw: bytes) -> pd.DataFrame:
    return dl.load_csv(raw)


# ---------------------------------------------------------------------------
# Data collection page
# ---------------------------------------------------------------------------

def _collect_raw() -> tuple[pd.DataFrame | None, bool]:
    """Sidebar source picker. Returns (raw dataframe, already_standardized)."""
    st.sidebar.header("📥 Data Source")
    choice = st.sidebar.radio("Choose a source", list(dl.DATA_SOURCES.keys()),
                              format_func=dl.DATA_SOURCES.get)

    if choice == "sample":
        st.sidebar.caption("600+ demo records - no API keys needed.")
        return _sample(), True

    if choice == "csv":
        up = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if up is None:
            st.info("Upload a CSV file in the sidebar to begin.")
            return None, False
        try:
            return _csv(up.getvalue()), False
        except ValueError as e:
            st.error(str(e))
            return None, False

    if choice == "amazon":
        st.sidebar.caption("Public datasets only, e.g. Amazon Reviews 2023 (McAuley Lab) "
                           "or Kaggle Amazon review CSVs.")
        path = st.sidebar.text_input("Dataset file path or URL (.csv / .jsonl)")
        limit = st.sidebar.number_input("Max rows", 100, 100_000, 5000, step=500)
        if st.sidebar.button("Load reviews") and path:
            try:
                st.session_state["collected"] = dl.load_amazon_reviews(path, int(limit))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load dataset: {e}")
        return st.session_state.get("collected"), True

    if choice == "social":
        platform = st.sidebar.selectbox("Platform", ["X / Twitter (API v2)", "Reddit (official API)"])
        query = st.sidebar.text_input("Search query", "iPhone")
        n = st.sidebar.slider("Max posts", 10, 100, 50)
        st.sidebar.caption("Needs your own API credentials set as environment variables or "
                           "Streamlit secrets (TWITTER_BEARER_TOKEN or REDDIT_CLIENT_ID / "
                           "REDDIT_CLIENT_SECRET).")
        if st.sidebar.button("Fetch posts"):
            try:
                fn = dl.fetch_twitter_posts if platform.startswith("X") else dl.fetch_reddit_posts
                st.session_state["collected"] = fn(query, n)
            except Exception as e:  # noqa: BLE001
                st.error(f"{e}  - Tip: use the sample dataset to try the app without keys.")
        return st.session_state.get("collected"), True

    if choice == "news":
        mode = st.sidebar.selectbox("Provider", ["Public RSS feed (no key)", "NewsAPI.org (key)"])
        if mode.startswith("Public"):
            url = st.sidebar.text_input("RSS feed URL", "https://feeds.bbci.co.uk/news/rss.xml")
            topic = st.sidebar.text_input("Topic label", "World News")
            if st.sidebar.button("Fetch headlines"):
                try:
                    st.session_state["collected"] = dl.fetch_news_rss(url, topic)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read feed: {e}")
        else:
            query = st.sidebar.text_input("Search query", "electric vehicles")
            if st.sidebar.button("Fetch articles"):
                try:
                    st.session_state["collected"] = dl.fetch_news_newsapi(query)
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
        return st.session_state.get("collected"), True

    return None, False


def _column_selection(raw: pd.DataFrame) -> pd.DataFrame | None:
    st.subheader("🧭 Column Selection")
    cols = list(raw.columns)
    guess = dl.suggest_mapping(raw)
    none = "— none —"

    def pick(label, role, required=False):
        options = cols if required else [none] + cols
        default = guess.get(role)
        idx = options.index(default) if default in options else 0
        val = st.selectbox(label, options, index=idx, key=f"map_{role}")
        return None if val == none else val

    c1, c2, c3 = st.columns(3)
    with c1:
        text = pick("Text column *", "text", required=True)
        date = pick("Date column", "date")
    with c2:
        source = pick("Source column", "source")
        product = pick("Product column", "product")
    with c3:
        rating = pick("Rating column", "rating")
    used = {text, date, source, product, rating}
    extra = st.multiselect("Additional columns to keep", [c for c in cols if c not in used],
                           default=[c for c in ["ID", "Sentiment", "Emotion"] if c in cols and c not in used])
    try:
        return dl.apply_mapping(raw, dl.ColumnMapping(text, date, source, product, rating, extra))
    except ValueError as e:
        st.error(str(e))
        return None


def data_page() -> None:
    st.title("💬 Sentiment Analysis Dashboard")
    st.caption("Dataset & data collection")

    raw, standardized = _collect_raw()
    if raw is None or raw.empty:
        if raw is not None:
            st.warning("The source returned no records.")
        return

    df = dl.standardize(raw) if standardized else _column_selection(raw)
    if df is None:
        return

    # Optional: restrict visible columns
    with st.expander("Choose columns to display / export"):
        visible = st.multiselect("Columns", list(df.columns), default=list(df.columns))
    view = df[visible] if visible else df

    # Dataset information
    st.subheader("ℹ️ Dataset Information")
    info = dl.dataset_info(df)
    m = st.columns(5)
    m[0].metric("Rows", f"{info['rows']:,}")
    m[1].metric("Columns", info["columns"])
    m[2].metric("Missing cells", info["missing_cells"])
    m[3].metric("Duplicate rows", info["duplicate_rows"])
    m[4].metric("Memory", f"{info['memory_kb']} KB")
    st.dataframe(dl.column_summary(df), width="stretch", hide_index=True)

    # Preview
    st.subheader("👀 Dataset Preview")
    n = st.slider("Rows to preview", 5, min(200, len(view)), min(20, len(view)))
    st.dataframe(view.head(n), width="stretch", hide_index=True)

    # Quality checks
    q1, q2 = st.columns(2)
    with q1:
        st.subheader("🕳️ Missing Values")
        mv = dl.missing_values_report(df)
        if mv.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(mv, width="stretch", hide_index=True)
    with q2:
        st.subheader("🧬 Duplicate Records")
        mode = st.radio("Compare by", ["Entire row", "Text only"], horizontal=True)
        subset = ["Text"] if mode == "Text only" else None
        dups = dl.detect_duplicates(df, subset=subset, keep=False)
        if dups.empty:
            st.success("No duplicates found.")
        else:
            st.warning(f"{len(dl.detect_duplicates(df, subset))} duplicate record(s) "
                       f"({len(dups)} rows involved).")
            st.dataframe(dups.sort_values("Text"), width="stretch", hide_index=True, height=240)

    # Cleaning
    st.subheader("🧹 Cleaning")
    c1, c2 = st.columns(2)
    drop_missing = c1.checkbox("Drop rows with empty text", value=True)
    drop_dups = c2.checkbox("Remove duplicate records", value=True)
    clean = df
    if drop_missing:
        clean = dl.handle_missing(clean, "drop_text")
    if drop_dups:
        clean = dl.remove_duplicates(clean, subset=subset)
    st.caption(f"{len(df):,} → {len(clean):,} rows after cleaning.")

    # Share with the rest of the dashboard
    st.session_state["df"] = clean
    st.session_state["df_raw"] = df

    # Download
    st.subheader("⬇️ Download Dataset")
    out = clean[[c for c in visible if c in clean.columns]] if visible else clean
    d1, d2, d3 = st.columns(3)
    d1.download_button("CSV", dl.to_csv_bytes(out), "sentiment_dataset.csv", "text/csv")
    d2.download_button("JSON", dl.to_json_bytes(out), "sentiment_dataset.json", "application/json")
    try:
        d3.download_button("Excel", dl.to_excel_bytes(out), "sentiment_dataset.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        d3.caption("Install openpyxl for Excel export.")


def overview_page() -> None:
    """Small example of another page consuming the shared dataset."""
    st.title("📊 Sentiment Overview")
    df = st.session_state.get("df")
    if df is None:
        st.info("Load a dataset on the Data page first.")
        return
    if df["Sentiment"].notna().any():
        c1, c2 = st.columns(2)
        c1.bar_chart(df["Sentiment"].value_counts())
        c2.bar_chart(df["Emotion"].value_counts())
    if df["Date"].notna().any() and df["Sentiment"].notna().any():
        trend = (df.dropna(subset=["Date"]).groupby([pd.Grouper(key="Date", freq="W"), "Sentiment"])
                 .size().unstack(fill_value=0))
        st.line_chart(trend)
    if df["Source"].notna().any():
        st.bar_chart(df["Source"].value_counts())


PAGES = {"Data": data_page, "Overview": overview_page}

if True:  # Streamlit runs this file as a script
    page = st.sidebar.selectbox("Page", list(PAGES.keys()))
    PAGES[page]()
