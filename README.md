# QianTick: Multimodal Deep Learning for Stock Price Direction Prediction

QianTick is a multimodal deep learning system that predicts next-day stock price direction by jointly learning from **financial news** (FinBERT embeddings), **historical price data** (LSTM / GRU / TCN), and **technical indicators**. A cross-attention fusion model combines both modalities in a dual-task (classification + regression) setup.

**Team — CSCI 566, USC, Spring 2026**

| Name | Email |
|------|-------|
| Ray Zhang | rzhang43@usc.edu |
| Kaicheng Li | kli32826@usc.edu |
| Hanzhen Qin | hanzhenq@usc.edu |
| Yunfei Wang | yunfeiwa@usc.edu |
| Yuxuan Qian | qianyuxu@usc.edu |
| Zeli Zhu | zelizhu@usc.edu |

---

## Pipeline Architecture

```
Raw News Headlines
    │
    ▼
[FinBERT Encoder]           → 771-dim per article (768 [CLS] + 3 sentiment)
    │
    ▼
[News Aggregator]           → 771-dim per (ticker, date)
    │
    └─────────────────────────────────────┐
                                          │
Raw OHLCV Data                           │
    │                                     │
    ▼                                     │
[Feature Engineering]       → 40 features/day (12 price + 28 technical indicators)
    │                                     │
    ▼                                     │
[LSTM / GRU / TCN]          → direction + return prediction (unimodal baseline)
    │                                     │
    ▼ (20 normalized tech features)       │
    └──────────────┬──────────────────────┘
                   ▼
        [Cross-Attention Fusion]
          price (Q) × news (K, V)
                   │
           ┌───────┴──────────┐
           ▼                  ▼
    [Classifier Head]   [Regression Head]
     P(up) via BCE       return via Huber
```

---

## Experimental Results

### Price Branch — Unimodal (Test set, n = 5,675 samples)

| Model | MAE | RMSE | R² | Direction Acc |
|-------|-----|------|----|---------------|
| LSTM  | 0.0175 | 0.0267 | −0.049 | **52.16%** |
| GRU   | 0.0168 | 0.0262 | −0.010 | 50.73% |
| TCN   | 0.0170 | 0.0263 | −0.018 | **52.19%** |
| Random baseline | — | — | — | 50.00% |

### Ablation: News-Only MLP (filtered test set, label threshold = 0.5%)

| Accuracy | ROC-AUC | Reg MAE | Reg RMSE |
|----------|---------|---------|----------|
| 50.89% | 50.78% | 0.0267 | 0.0365 |

### Fusion Model: Cross-Attention News + Price (same filtered test set)

| Metric | News-Only | Fusion |
|--------|-----------|--------|
| Accuracy | 50.89% | 50.53% |
| ROC-AUC | 50.78% | 50.38% |
| Reg MAE | 0.0267 | 0.0301 |
| Reg RMSE | 0.0365 | 0.0413 |

All models beat the 50% random baseline. Near-zero R² is consistent with the Efficient Market Hypothesis — even professional funds rarely exceed ~55% direction accuracy.

---

## Repository Structure

```
QianTick/
├── configs/
│   ├── news_config.yaml          # Hyperparameters for news pipeline
│   └── fusion_config.yaml        # Hyperparameters for fusion model
├── data/
│   ├── processed/news/           # Raw CSVs (Finnhub, GDELT)
│   └── features/                 # Generated parquet feature files
├── documents/                    # Internal task specs and project summary
├── export_to_fusion/
│   ├── features/                 # Price model hidden states (.pt)
│   └── predictions/              # Price model outputs CSV
├── outputs/
│   └── fusion/                   # Fusion checkpoints, metrics, predictions
├── scripts/
│   └── eval_price_on_fusion_testset.py
├── src/
│   ├── data_collection/
│   │   ├── finnhub_collector.py
│   │   ├── gdelt_collector.py
│   │   ├── merge_finnhub_gdelt.py
│   │   └── team_news_collector_incremental.py
│   ├── news/
│   │   ├── preprocess.py         # Raw CSV → news_cleaned.parquet
│   │   ├── encoder.py            # FinBERT article embeddings (771-dim)
│   │   ├── aggregator.py         # Per-(ticker, date) feature aggregation
│   │   ├── feature_extractor.py
│   │   ├── dataset.py            # PyTorch Dataset for news features
│   │   ├── model.py              # News unimodal MLP baseline
│   │   └── train.py
│   ├── price_prediction_pipeline/
│   │   ├── fetch_data.py         # OHLCV data via yfinance
│   │   ├── feature_engineering.py  # 40 technical features
│   │   ├── dataset_builder.py    # Sliding-window dataset construction
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── tcn_model.py
│   │   ├── metrics.py
│   │   ├── fusion_interface.py   # Exports hidden states for fusion
│   │   └── main_pipeline.py      # Optuna tuning + full training run
│   └── fusion/
│       ├── dataset.py            # Aligns news + price features by (ticker, date)
│       ├── model.py              # Cross-Attention QianTickModel
│       └── train.py
├── Submitted_Reports/
│   ├── Proj Proposal.pdf
│   ├── CSCI_566_Term_Proj_MidReport.pdf
│   └── CSCI_566_Term_Proj_Final_Report.pdf
├── tests/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.11, PyTorch 2.8, Transformers 4.57, Optuna 4.8, pandas 2.3, pyarrow 21.

Copy `.env.example` to `.env` and fill in your Finnhub API key before running data collection.

---

## Running the Pipeline

**1. Collect news data**
```bash
python -m src.data_collection.finnhub_collector
python -m src.data_collection.gdelt_collector
python -m src.data_collection.merge_finnhub_gdelt
```

**2. News preprocessing → embeddings → aggregation**
```bash
python -m src.news.preprocess    --config configs/news_config.yaml
python -m src.news.encoder       --config configs/news_config.yaml
python -m src.news.aggregator    --config configs/news_config.yaml
```

**3. Price pipeline (feature engineering + Optuna tuning + training)**
```bash
python -m src.price_prediction_pipeline.main_pipeline
```

**4. Fusion model**
```bash
python -m src.fusion.train --config configs/fusion_config.yaml
```

**5. Evaluate price models on fusion test set**
```bash
python scripts/eval_price_on_fusion_testset.py
```

---

## Key Design Decisions

| Decision | Detail |
|----------|--------|
| Ticker format | Uppercase string (`AAPL`, `NVDA`, …) |
| Date format | `YYYY-MM-DD` |
| News feature dim | 771 (`feat_0` ~ `feat_770`) — FinBERT 768 + 3 sentiment |
| Price feature dim | 40 features/day; 20 used as fusion input |
| File format | `.parquet` for news/fusion features |
| Train / Val / Test split | ≤ 2024-06-30 / 2024-07-01 ~ 2024-12-31 / ≥ 2025-01-01 |
| Random seed | 42 |
| Timezone cutoff | Articles after 4:00 PM ET → next NYSE trading day |

---

## Submitted Reports

| Report | File |
|--------|------|
| Project Proposal | `Submitted_Reports/Proj Proposal.pdf` |
| Mid-term Report | `Submitted_Reports/CSCI_566_Term_Proj_MidReport.pdf` |
| Final Report | `Submitted_Reports/CSCI_566_Term_Proj_Final_Report.pdf` |

---

© 2026 Ray Zhang, Kaicheng Li, Hanzhen Qin, Yunfei Wang, Yuxuan Qian, Zeli Zhu. All rights reserved.
