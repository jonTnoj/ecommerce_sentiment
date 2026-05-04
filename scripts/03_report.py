"""Stage 3: generate a static, self-contained HTML report.

Reads the scored CSV from stage 2, builds every Plotly figure, and writes
a single output/report.html file plus PNG word clouds.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.io as pio  # noqa: E402

from src import config, viz  # noqa: E402


def _load_scored_df() -> pd.DataFrame:
    if not config.SCORED_CSV.exists():
        raise FileNotFoundError(
            f"{config.SCORED_CSV} not found. Run scripts/02_score.py first."
        )
    df = pd.read_csv(config.SCORED_CSV)
    df["aspects"] = df["aspects"].apply(json.loads)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
    return df


def _save_wordclouds(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One PNG per polarity, plus one per (negative, aspect) combination."""
    saved = []
    for polarity in ("negative", "positive"):
        wc = viz.wordcloud_for(df, polarity=polarity)
        if wc is None:
            continue
        path = out_dir / f"wordcloud_{polarity}.png"
        wc.to_file(str(path))
        saved.append(path)

    # Per-aspect negative wordclouds — the most actionable view for complaints.
    for aspect in config.ASPECT_KEYWORDS:
        wc = viz.wordcloud_for(df, polarity="negative", aspect=aspect)
        if wc is None:
            continue
        slug = aspect.lower().replace(" ", "_").replace("/", "")
        path = out_dir / f"wordcloud_negative_{slug}.png"
        wc.to_file(str(path))
        saved.append(path)

    return saved


def _metrics_html() -> str:
    """Render output/metrics.json as an HTML summary block."""
    if not config.METRICS_JSON.exists():
        return "<p><i>No metrics.json found.</i></p>"
    metrics = json.loads(config.METRICS_JSON.read_text())
    cls = metrics["classification"]
    reg = metrics["regression"]

    rows = [
        ("Total reviews scored", f"{metrics['n_reviews']:,}"),
        ("MAE (predicted stars vs actual)", f"{reg['mae']:.3f}"),
        ("RMSE", f"{reg['rmse']:.3f}"),
        ("3-class polarity accuracy", f"{cls['accuracy']:.3f}"),
        ("Macro F1", f"{cls['macro_avg']['f1-score']:.3f}"),
        ("Weighted F1", f"{cls['weighted_avg']['f1-score']:.3f}"),
    ]
    summary_table = "<table class='metrics'>" + "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows
    ) + "</table>"

    per_class_rows = "".join(
        f"<tr><th>{label}</th>"
        f"<td>{cls['per_class'][label]['precision']:.3f}</td>"
        f"<td>{cls['per_class'][label]['recall']:.3f}</td>"
        f"<td>{cls['per_class'][label]['f1-score']:.3f}</td>"
        f"<td>{int(cls['per_class'][label]['support']):,}</td></tr>"
        for label in cls["labels"]
    )
    per_class = (
        "<table class='metrics'><thead><tr>"
        "<th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>"
        "</tr></thead><tbody>" + per_class_rows + "</tbody></table>"
    )
    return summary_table + "<h3>Per-class breakdown</h3>" + per_class


def build_report() -> Path:
    df = _load_scored_df()
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("Star Rating Distribution",     viz.fig_rating_distribution(df)),
        ("VADER Polarity Distribution",  viz.fig_polarity_distribution(df)),
        ("Confusion Matrix",             viz.fig_confusion_matrix(df)),
        ("Sentiment vs Rating",          viz.fig_sentiment_vs_rating(df)),
        ("Aspect Mention Volume",        viz.fig_aspect_volume(df)),
        ("Aspect × Polarity Heatmap",   viz.fig_aspect_polarity_heatmap(df)),
        ("Sentiment Over Time",          viz.fig_sentiment_over_time(df)),
        ("Top Complaint Products",       viz.fig_top_complaint_products(df)),
        ("Verified vs Unverified",       viz.fig_verified_vs_not(df)),
    ]

    # Only include Plotly.js once — in the first figure block.
    first_title = figures[0][0]
    fig_blocks = []
    for title, fig in figures:
        fig_html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if title == first_title else False,
            full_html=False,
        )
        fig_blocks.append(f"<section><h2>{title}</h2>{fig_html}</section>")

    cloud_paths = _save_wordclouds(df, out_dir)
    cloud_html = "".join(
        f"<figure><img src='{p.name}' alt='{p.stem}'>"
        f"<figcaption>{p.stem.replace('_', ' ').title()}</figcaption></figure>"
        for p in cloud_paths
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E-commerce Review Sentiment Analysis -- Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif;
          max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1, h2, h3 {{ color: #1a3a5c; }}
  section {{ margin: 2em 0; padding-top: 1em; border-top: 1px solid #eee; }}
  table.metrics {{ border-collapse: collapse; margin: 1em 0; }}
  table.metrics th, table.metrics td {{
      padding: 6px 12px; border: 1px solid #ddd; text-align: left; }}
  table.metrics th {{ background: #f3f6fa; }}
  figure {{ display: inline-block; margin: 0.5em; vertical-align: top; }}
  figure img {{ max-width: 460px; border: 1px solid #ddd; }}
  figcaption {{ font-size: 0.85em; color: #555; text-align: center; }}
  .header-meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>E-commerce Review Sentiment Analysis</h1>
<p class="header-meta">CS 210 final-project report — generated by
<code>scripts/03_report.py</code> from <code>output/scored_reviews.csv</code>.</p>

<section>
<h2>Headline Metrics</h2>
{_metrics_html()}
</section>

{"".join(fig_blocks)}

<section>
<h2>Word Clouds</h2>
<p>Most-frequent terms within negatively-scored reviews (overall and per
aspect category) — what people are actually complaining about.</p>
{cloud_html}
</section>

</body></html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html)
    return report_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report_path = build_report()
    print(f"\nReport written to: {report_path}")
