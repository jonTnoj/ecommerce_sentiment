"""
Plotly figure builders.

Each function takes a scored DataFrame and returns a ``plotly.graph_objects.Figure``.
Both the Streamlit dashboard and the static-HTML report import these, so any
visual change made here updates both surfaces.

Style note: we keep colors consistent across charts -- negative is red,
neutral is gray, positive is green -- so the reader can scan multiple
charts without re-decoding the palette each time.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config

# Shared palette. Same colors across every chart.
POLARITY_COLORS = {
    "negative": "#d62728",  # red
    "neutral": "#7f7f7f",   # gray
    "positive": "#2ca02c",  # green
}

POLARITY_ORDER = ["negative", "neutral", "positive"]


# ------------------------------------------------------------------ basics


def fig_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of star-rating counts. Establishes class imbalance up front."""
    counts = df["rating"].value_counts().sort_index()
    fig = px.bar(
        x=counts.index.astype(int),
        y=counts.values,
        labels={"x": "Star rating", "y": "Reviews"},
        title="Star Rating Distribution",
        text=counts.values,
    )
    fig.update_traces(textposition="outside", marker_color="#1f77b4")
    fig.update_layout(showlegend=False)
    return fig


def fig_polarity_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of VADER-predicted polarity counts."""
    counts = df["predicted_polarity"].value_counts().reindex(POLARITY_ORDER, fill_value=0)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=POLARITY_COLORS,
        labels={"x": "Predicted polarity", "y": "Reviews"},
        title="VADER Predicted Polarity Distribution",
        text=counts.values,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# ------------------------------------------------------------------ agreement


def fig_confusion_matrix(df: pd.DataFrame) -> go.Figure:
    """Heatmap: predicted polarity vs star-bucketed actual polarity."""
    labels = POLARITY_ORDER
    cm = (
        pd.crosstab(df["actual_polarity"], df["predicted_polarity"])
        .reindex(index=labels, columns=labels, fill_value=0)
    )
    # Row-normalize so we read it as "of all reviews actually X, what fraction
    # were predicted Y" -- much more useful than raw counts when classes are
    # imbalanced.
    cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    annotations = [
        [f"{cm.iloc[i,j]:,}<br>{cm_norm.iloc[i,j]:.1%}" for j in range(len(labels))]
        for i in range(len(labels))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_norm.values,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=annotations,
            texttemplate="%{text}",
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{text}<extra></extra>",
            colorbar=dict(title="Row<br>share"),
        )
    )
    fig.update_layout(
        title="Confusion Matrix: Star-bucketed Actual vs VADER Predicted",
        xaxis_title="Predicted polarity (VADER)",
        yaxis_title="Actual polarity (from star rating)",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_sentiment_vs_rating(df: pd.DataFrame, max_points: int = 8000) -> go.Figure:
    """Scatter: VADER compound score against actual star rating.

    Includes a small horizontal jitter on the x (ratings are integer) to
    avoid every point landing on five vertical lines. Sub-samples for
    rendering speed.
    """
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.18, 0.18, size=len(df))
    fig = px.scatter(
        x=df["rating"] + jitter,
        y=df["compound"],
        color=df["predicted_polarity"],
        color_discrete_map=POLARITY_COLORS,
        category_orders={"color": POLARITY_ORDER},
        opacity=0.45,
        labels={"x": "Star rating (jittered)", "y": "VADER compound score", "color": "Polarity"},
        title="VADER Compound Score vs Star Rating",
    )
    # Reference line: perfect linear mapping rating <-> compound.
    fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.4)
    return fig


# ------------------------------------------------------------------ aspects


def fig_aspect_polarity_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of aspect category × predicted polarity, normalized per-aspect.

    Reading: a row tells you, of the reviews that mentioned <aspect>, what
    fraction were positive / neutral / negative. Useful for spotting
    aspects that disproportionately attract complaints.
    """
    rows = []
    for aspect in config.ASPECT_KEYWORDS:
        mask = df["aspects"].apply(lambda al, a=aspect: a in al)
        sub = df[mask]
        if len(sub) == 0:
            rows.append((aspect, 0, 0, 0, 0))
            continue
        d = sub["predicted_polarity"].value_counts(normalize=True)
        rows.append(
            (
                aspect,
                int(len(sub)),
                float(d.get("negative", 0)),
                float(d.get("neutral", 0)),
                float(d.get("positive", 0)),
            )
        )
    adf = pd.DataFrame(rows, columns=["aspect", "n", "negative", "neutral", "positive"])
    z = adf[["negative", "neutral", "positive"]].values
    text = [
        [f"{adf.iloc[i]['n']:,} reviews<br>{z[i][j]:.1%}" for j in range(3)]
        for i in range(len(adf))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=POLARITY_ORDER,
            y=adf["aspect"],
            colorscale="RdYlGn",
            zmid=0.33,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Share"),
        )
    )
    fig.update_layout(
        title="Aspect × Polarity Heatmap (row-normalized)",
        xaxis_title="Predicted polarity",
        yaxis_title="Aspect category",
        height=500,
    )
    return fig


def fig_aspect_volume(df: pd.DataFrame) -> go.Figure:
    """Bar chart of how many reviews mentioned each aspect."""
    counts = {
        aspect: int(df["aspects"].apply(lambda al, a=aspect: a in al).sum())
        for aspect in config.ASPECT_KEYWORDS
    }
    s = pd.Series(counts).sort_values(ascending=True)
    fig = px.bar(
        x=s.values,
        y=s.index,
        orientation="h",
        labels={"x": "Reviews mentioning aspect", "y": "Aspect"},
        title="Aspect Mention Volume",
        text=s.values,
    )
    fig.update_traces(textposition="outside", marker_color="#1f77b4")
    return fig


# ------------------------------------------------------------------ time


def fig_sentiment_over_time(df: pd.DataFrame) -> go.Figure:
    """Monthly average VADER compound score with review volume on a 2nd axis."""
    monthly = (
        df.groupby("year_month")
        .agg(avg_compound=("compound", "mean"), n=("compound", "size"))
        .reset_index()
        .sort_values("year_month")
    )
    # Drop any month with very few reviews -- noisy and clutters the chart.
    monthly = monthly[monthly["n"] >= 30]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["year_month"],
            y=monthly["avg_compound"],
            mode="lines+markers",
            name="Avg compound",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly["year_month"],
            y=monthly["n"],
            name="Reviews",
            opacity=0.25,
            yaxis="y2",
            marker_color="#888",
        )
    )
    fig.update_layout(
        title="Average Sentiment Over Time (months with >=30 reviews)",
        xaxis_title="Month",
        yaxis=dict(title="Avg VADER compound", side="left"),
        yaxis2=dict(title="Review count", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ------------------------------------------------------------------ products


def fig_top_complaint_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top-N products by negative-review count (must have >= 5 reviews)."""
    by_product = (
        df.groupby("parent_asin")
        .agg(
            n=("rating", "size"),
            n_negative=("predicted_polarity", lambda s: (s == "negative").sum()),
            avg_compound=("compound", "mean"),
        )
        .reset_index()
    )
    by_product = by_product[by_product["n"] >= 5]
    by_product = by_product.sort_values("n_negative", ascending=False).head(top_n)
    by_product["label"] = by_product["parent_asin"] + " (n=" + by_product["n"].astype(str) + ")"

    fig = px.bar(
        by_product[::-1],
        x="n_negative",
        y="label",
        orientation="h",
        title=f"Top {top_n} Products by Negative-Review Volume (min 5 reviews)",
        labels={"n_negative": "Negative reviews", "label": "Parent ASIN"},
        color="avg_compound",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text="n_negative",
    )
    fig.update_traces(textposition="outside")
    return fig


# ------------------------------------------------------------------ verified


def fig_verified_vs_not(df: pd.DataFrame) -> go.Figure:
    """Box plot: VADER score split by verified vs unverified purchase."""
    fig = px.box(
        df,
        x="verified_purchase",
        y="compound",
        color="verified_purchase",
        title="VADER Compound Score: Verified vs Unverified Purchases",
        labels={"verified_purchase": "Verified purchase", "compound": "VADER compound"},
        points=False,
    )
    fig.update_layout(showlegend=False)
    return fig


# ------------------------------------------------------------------ word clouds


def wordcloud_for(
    df: pd.DataFrame,
    polarity: str = "negative",
    aspect: str | None = None,
    max_words: int = 100,
):
    """Generate a wordcloud (returns matplotlib Figure, not plotly).

    The wordcloud package draws PIL images directly. The Streamlit dashboard
    embeds these as PNGs.
    """
    from wordcloud import STOPWORDS, WordCloud

    sub = df[df["predicted_polarity"] == polarity]
    if aspect:
        sub = sub[sub["aspects"].apply(lambda al, a=aspect: a in al)]
    if len(sub) == 0:
        return None
    text = " ".join(sub["text"].fillna("").astype(str).tolist())
    if not text.strip():
        return None
    stopwords = set(STOPWORDS) | {"product", "one", "thing", "got", "amazon", "br"}
    wc = WordCloud(
        width=900,
        height=420,
        background_color="white",
        max_words=max_words,
        stopwords=stopwords,
        colormap="Reds" if polarity == "negative" else "Greens",
    ).generate(text)
    return wc
