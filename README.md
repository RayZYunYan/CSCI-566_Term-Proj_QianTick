# QianTick: Multimodal Deep Learning for Stock Price Direction Prediction

QianTick is a multimodal deep learning system designed for short-term stock price direction prediction. The project integrates **historical price data**, **financial news**, and **technical indicators** to model complex market dynamics using modern neural architectures and fusion strategies. The system follows industry best practices for machine learning engineering, emphasizing **reproducibility**, **modularity**, and **scalability**.

**Team:**
- Ray Zhang: rzhang43@usc.edu
- Zeli Zhu: zelizhu@usc.edu
- Kaicheng Li: kli32826@usc.edu
- Hanzhen Qin: hanzhenq@usc.edu
- Yunfei Wang: yunfeiwa@usc.edu
- Yuxuan Qian: qianyuxu@usc.edu

---

## Project Motivation

Stock price movements are influenced by multiple heterogeneous information sources:
- **Temporal price patterns** (market dynamics)
- **Textual information** (news events and sentiment)
- **Technical indicators** (handcrafted financial signals)

Most traditional approaches rely on a single modality, limiting their ability to capture cross-source interactions. QianTick addresses this limitation by proposing a **multimodal fusion framework** that jointly learns from structured time-series data and unstructured textual data.

---

## Repository Structure

```
QianTick/
├── configs/
│   ├── news_config.yaml          # Central config for the news module (all hyperparameters)
│   └── fusion_config.yaml
├── data/
│   ├── processed/news/
│   │   ├── finnhub_merged.csv    # Raw per-ticker news (Finnhub)
│   │   ├── gdelt_merged.csv      # Raw market-wide tech news (GDELT)
│   │   └── news_cleaned.parquet  # [Sub-task A output] cleaned news
│   └── features/
│       ├── news_article_embeddings.parquet  # [Sub-task B output]
│       ├── news_features.parquet            # [Sub-task C output]
│       └── labels.parquet
├── scripts/
│   └── news_e2e_test.py          # End-to-end integration test (A→B→C→D)
├── src/
│   ├── news/
│   │   ├── preprocess.py         # [Sub-task A] raw CSV → news_cleaned.parquet
│   │   ├── encoder.py            # [Sub-task B] FinBERT article embeddings
│   │   ├── aggregator.py         # [Sub-task C] per-(ticker,date) feature aggregation
│   │   ├── dataset.py            # [Sub-task D] PyTorch Dataset
│   │   ├── model.py              # [Sub-task D] unimodal baseline model
│   │   └── train.py              # [Sub-task D] training loop
│   ├── fusion/                   # Multimodal fusion module
│   └── price_prediction_pipeline/
├── tests/
└── requirements.txt
```

---

## Shared Conventions

All subtasks read from `configs/news_config.yaml`. Do **not** hardcode these values in Python files.

| Convention | Value |
|---|---|
| Ticker format | Uppercase string (`AAPL`, `NVDA`, …) |
| Date format | `YYYY-MM-DD` string |
| Feature dimension | 771 (`feat_0` ~ `feat_770`) |
| File format | `.parquet` |
| Config | `configs/news_config.yaml` |
| Random seed | 42 |
| Train split | ≤ `2024-06-30` |
| Val split | `2024-07-01` ~ `2024-12-31` |
| Test split | ≥ `2025-01-01` |

---

## News Module Progress

### Sub-task A — Data Preprocessing ✅ Complete

**Owner:** Ray Zhang

**Deliverables:**

| File | Description |
|---|---|
| `src/news/preprocess.py` | Cleans raw CSVs → `news_cleaned.parquet` |
| `configs/news_config.yaml` | Central config for all news subtasks |
| `scripts/news_e2e_test.py` | A→B→C→D integration test |

**Output schema** (`news_cleaned.parquet`):

| Column | Type | Rule |
|---|---|---|
| `ticker` | str | Uppercase (`AAPL`, `TSLA`, …) |
| `date` | str | `YYYY-MM-DD` — prediction date |
| `article_id` | str | Unique article identifier |
| `headline` | str | Cleaned text |
| `publish_datetime` | datetime | Original publish time (UTC) |

**Processing rules applied:**
- **Timezone cutoff**: articles published after 4:00 PM ET → assigned to next NYSE trading day
- **Similarity dedup**: normalized headline (lowercase + strip punctuation) hash per `(ticker, headline)` — keeps earliest occurrence
- **Headline filter**: removes null, length < 10, and >30% non-ASCII characters
- **7-day window**: each article on day T appears in prediction dates T ~ T+6 (trading days only)
- **Trading days**: NYSE approximate calendar — US federal holidays + Good Friday; weekends excluded
- **GDELT**: market-wide tech news (no ticker in source) is broadcast to all 15 tickers

**Output stats:**
- ~10.8M rows, 15 tickers, date range 2020-06-11 ~ 2026-02-26
- Zero weekend dates in output

**Run preprocessing:**
```bash
python -m src.news.preprocess --config configs/news_config.yaml
```

**Run integration test:**
```bash
# Step A uses real CSVs; Steps B/C/D fall back to mock if not yet implemented
python scripts/news_e2e_test.py --mock-bcd
```

---

### Sub-task B — Article Embedding 🔲 Pending

`src/news/encoder.py` — FinBERT encoder → `data/features/news_article_embeddings.parquet`

**Input:** `news_cleaned.parquet`  
**Output schema:** `ticker`, `date`, `article_id`, `feat_0` ~ `feat_770` (771-dim)

---

### Sub-task C — Feature Aggregation 🔲 Pending

`src/news/aggregator.py` — per-(ticker, date) aggregation → `data/features/news_features.parquet`

**Input:** `news_article_embeddings.parquet`  
**Output schema:** flat columns `ticker`, `date`, `feat_0` ~ `feat_770`

---

### Sub-task D — Model Training 🔲 Pending

`src/news/dataset.py`, `src/news/model.py`, `src/news/train.py`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** Python 3.11+, PyTorch ≥ 2.1, Transformers ≥ 4.38, pandas ≥ 2.0, pyarrow ≥ 15.0
