# E-commerce Review Sentiment Analysis

CS 210 — Data Management for Data Science final project.

A "feedback sorter" pipeline that ingests Amazon product reviews into a
NoSQL document store, runs lexicon-based sentiment analysis, tags each review
with one or more *aspect categories* (Product Quality, Shipping, Price, …),
and surfaces the results through both a static HTML report and an
interactive Streamlit dashboard.

## What it does

1. **Ingest.** Streams a gzipped JSONL of ~2.13M Amazon Appliances reviews
   into MongoDB. Uses reservoir sampling stratified by star rating so we
   never load the full file into memory and end up with a balanced
   50,000-row sample by default.
2. **Score.** Pulls the documents from the store, runs VADER sentiment
   on each review's title + body, and tags aspects with a keyword lexicon.
3. **Evaluate.** Reports both regression metrics (MAE / RMSE between
   predicted-from-text stars and actual rating) and classification metrics
   (precision / recall / F1 for the 3-class polarity task).
4. **Visualize.** Builds 9 interactive Plotly charts plus per-aspect
   word clouds, served in a Streamlit dashboard or compiled into one
   self-contained HTML file.

## Dataset

**Amazon Reviews 2023** (McAuley Lab, UC San Diego), Appliances category.
Citation:

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).
> Bridging Language and Items for Retrieval and Recommendation. *arXiv:2403.03952*.

Download: <https://amazon-reviews-2023.github.io/>

Place the file `Appliances_jsonl.gz` in `./data/` before running.

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. (Optional but recommended) Start MongoDB locally
docker compose up -d
# Or, if you don't have Docker, install MongoDB Community Edition:
#   https://www.mongodb.com/docs/manual/installation/
# If neither is available, the pipeline automatically falls back to
# TinyDB (a pure-Python file-based document store).

# 3. Drop the data file in place
mv Appliances_jsonl.gz data/

# 4. Run the pipeline
python scripts/01_ingest.py     # ~30s for sampling + insert
python scripts/02_score.py      # ~15s for VADER on 50k reviews
python scripts/03_report.py     # generates output/report.html

# 5. (Optional) Launch the interactive dashboard
streamlit run dashboard/app.py
```

End-to-end one-shot:

```bash
python scripts/run_pipeline.py
```

## Project layout

```
ecommerce_sentiment/
├── README.md
├── PROPOSAL_MODIFICATIONS.md   ← changes from the original proposal
├── REPORT_DRAFT.md             ← starting draft for the final writeup
├── requirements.txt
├── docker-compose.yml          ← one-command MongoDB
├── data/                       ← raw .jsonl.gz lives here (gitignored)
├── output/                     ← generated CSV / HTML / PNG / JSON
├── src/
│   ├── config.py               ← all tunables, aspect lexicons
│   ├── db.py                   ← Mongo + TinyDB abstraction
│   ├── ingest.py               ← stratified reservoir sampler
│   ├── preprocess.py           ← light text cleaning
│   ├── sentiment.py            ← VADER wrapper, polarity buckets
│   ├── categorize.py           ← keyword-based aspect tagging
│   ├── analyze.py              ← scoring + metrics orchestration
│   └── viz.py                  ← Plotly figure builders
├── scripts/
│   ├── 01_ingest.py
│   ├── 02_score.py
│   ├── 03_report.py
│   └── run_pipeline.py         ← all three stages, in order
├── dashboard/
│   └── app.py                  ← Streamlit dashboard
└── tests/
    └── test_smoke.py           ← unit-level sanity checks
```

## Reproducibility

The sample drawn at ingest time is deterministic given a fixed seed
(`config.RANDOM_SEED`, default 42). Re-running the pipeline produces
identical metrics down to the digit.

## Tunables

Everything lives in [`src/config.py`](src/config.py):

| Setting               | Default                | Notes                                              |
|-----------------------|------------------------|----------------------------------------------------|
| `SAMPLE_SIZE`         | 50,000                 | Total reviews to keep (10k per star).              |
| `RANDOM_SEED`         | 42                     | For the reservoir sampler.                         |
| `DB_BACKEND`          | `"mongo"`              | `"mongo_strict"` to fail loudly without MongoDB.   |
| `MONGO_URI`           | `mongodb://localhost`  | Atlas free tier works too — paste the URI here.    |
| `ASPECT_KEYWORDS`     | 6 categories           | Edit to extend the "feedback sorter" lexicon.      |

## Why MongoDB

Amazon reviews are unstructured text with optional fields (some have
images, some don't, helpful-vote count varies, titles are sometimes
blank). A document store handles this without schema migrations and
without forcing us to flatten the `images` array into a join table.
The proposal's database choice is justified on these grounds; see
[`PROPOSAL_MODIFICATIONS.md`](PROPOSAL_MODIFICATIONS.md) item 7 for
discussion of the TinyDB fallback.

Indexes created (see `db.py::create_indexes`):

* Compound `(parent_asin, timestamp)` — for the per-product
  time-series queries the dashboard does on the "Top Complaint
  Products" panel.
* Single field `rating` — for the rating filter.
* Single field `predicted_polarity` — for the polarity filter.

## Limitations

These are spelled out in detail in `REPORT_DRAFT.md`. The short version:

* **VADER is lexicon-based**, so it misses sarcasm and complex negation.
* **Class imbalance** in the raw data (~70% 5-star) is masked by our
  stratified sampling. Real-world deployment would face it head-on.
* **Aspect categorization is keyword-based**, not learned. We have no
  human-labeled aspect ground truth to evaluate it against.
* **Single product category.** All results come from Appliances; do not
  generalize to e.g. Books without re-validating.

## Tests

```bash
python tests/test_smoke.py
# or
pytest tests/
```

## License

Educational use; no warranty.
