"""
Interactive dashboard.

Loads the scored CSV produced by ``scripts/02_score.py`` and exposes
filters (date range, star rating, polarity, aspect, verified flag) plus
all of the visualizations from ``src.viz``. Reactive: every chart updates
when you move a slider or toggle a checkbox.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when running via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, viz  # noqa: E402


st.set_page_config(
    page_title="E-commerce Review Sentiment Dashboard",
    layout="wide",
    page_icon="📊",
)


# --------------------------------------------------------------- data loader


@st.cache_data(show_spinner="Loading scored reviews...")
def load_data() -> pd.DataFrame:
    """Read the scored CSV. Cached so filter changes don't re-parse the file."""
    if not config.SCORED_CSV.exists():
        st.error(
            f"Could not find `{config.SCORED_CSV}`.\n\n"
            "Run the pipeline first:\n"
            "```\npython scripts/01_ingest.py\npython scripts/02_score.py\n```"
        )
        st.stop()
    df = pd.read_csv(config.SCORED_CSV)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
    df["date"] = df["datetime"].dt.date
    return df


df = load_data()


# --------------------------------------------------------------- sidebar filters

st.sidebar.title("Filters")

min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_date, max_date

ratings = st.sidebar.multiselect(
    "Star rating",
    options=[1.0, 2.0, 3.0, 4.0, 5.0],
    default=[1.0, 2.0, 3.0, 4.0, 5.0],
)

polarities = st.sidebar.multiselect(
    "Predicted polarity",
    options=["negative", "neutral", "positive"],
    default=["negative", "neutral", "positive"],
)

aspect_filter = st.sidebar.multiselect(
    "Must mention aspect (any of)",
    options=list(config.ASPECT_KEYWORDS.keys()),
    default=[],
)

verified_only = st.sidebar.checkbox("Verified purchases only", value=False)


# --------------------------------------------------------------- apply filters

mask = (
    (df["date"] >= start)
    & (df["date"] <= end)
    & (df["rating"].isin(ratings))
    & (df["predicted_polarity"].isin(polarities))
)
if verified_only:
    mask &= df["verified_purchase"]
if aspect_filter:
    mask &= df["aspects"].apply(
        lambda al: any(a in al for a in aspect_filter)
    )

filtered = df[mask].copy()


# --------------------------------------------------------------- header

st.title("📊 E-commerce Review Sentiment Dashboard")
st.markdown(
    "**Dataset:** Amazon Reviews 2023 -- Appliances category "
    "(McAuley Lab, UC San Diego)."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews (filtered)", f"{len(filtered):,}", f"of {len(df):,}")
c2.metric(
    "Avg. star rating",
    f"{filtered['rating'].mean():.2f}" if len(filtered) else "—",
)
c3.metric(
    "Avg. VADER compound",
    f"{filtered['compound'].mean():+.3f}" if len(filtered) else "—",
)
neg_share = (filtered["predicted_polarity"] == "negative").mean() if len(filtered) else 0
c4.metric("Share negative", f"{neg_share:.1%}")

if len(filtered) == 0:
    st.warning("No reviews match the current filter combination.")
    st.stop()


# --------------------------------------------------------------- main charts

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Model Agreement", "Aspects", "Drilldown"]
)

with tab1:
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_rating_distribution(filtered), use_container_width=True)
    b.plotly_chart(viz.fig_polarity_distribution(filtered), use_container_width=True)
    st.plotly_chart(viz.fig_sentiment_over_time(filtered), use_container_width=True)
    st.plotly_chart(viz.fig_verified_vs_not(filtered), use_container_width=True)

with tab2:
    st.markdown(
        "Two complementary framings of model performance. Reading the matrix "
        "row-by-row tells you, of reviews whose star rating maps to *Actual=X*, "
        "what fraction VADER assigned to each predicted polarity."
    )
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_confusion_matrix(filtered), use_container_width=True)
    b.plotly_chart(viz.fig_sentiment_vs_rating(filtered), use_container_width=True)

with tab3:
    a, b = st.columns(2)
    a.plotly_chart(viz.fig_aspect_volume(filtered), use_container_width=True)
    b.plotly_chart(viz.fig_aspect_polarity_heatmap(filtered), use_container_width=True)
    st.plotly_chart(viz.fig_top_complaint_products(filtered), use_container_width=True)

    st.markdown("### Word cloud: negative reviews")
    aspect_for_cloud = st.selectbox(
        "Filter cloud to aspect",
        options=["(any)"] + list(config.ASPECT_KEYWORDS.keys()),
    )
    aspect_arg = None if aspect_for_cloud == "(any)" else aspect_for_cloud
    wc = viz.wordcloud_for(filtered, polarity="negative", aspect=aspect_arg)
    if wc is None:
        st.info("Not enough negative reviews to draw a cloud for this filter.")
    else:
        st.image(wc.to_array(), use_container_width=True)

with tab4:
    st.markdown(
        "Sample of reviews matching the current filters. Useful for "
        "spot-checking why VADER assigned the polarity it did, and what "
        "real complaints look like in the data."
    )
    cols_to_show = [
        "datetime",
        "rating",
        "predicted_polarity",
        "compound",
        "aspects",
        "title",
        "text",
    ]
    sample_n = st.slider("Rows", min_value=10, max_value=200, value=50, step=10)
    st.dataframe(
        filtered[cols_to_show].sample(min(sample_n, len(filtered)), random_state=0),
        use_container_width=True,
        height=520,
    )

st.markdown("---")
st.caption(
    "Built for CS 210: Data Management for Data Science. "
    "Sentiment via VADER (Hutto & Gilbert, 2014). "
    "Storage: MongoDB (with TinyDB fallback)."
)
