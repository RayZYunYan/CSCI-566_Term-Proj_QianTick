"""
Evaluate LSTM / GRU / TCN on the fusion-compatible test set.

Fusion test set definition:
  - Tickers: 13 stocks with news coverage (no ETFs)
  - Date range: 2025-01-23 to 2026-02-26
  - Derived by intersecting the price pipeline test split with news_features.parquet

Run from repo root:
    python scripts/eval_price_on_fusion_testset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "price_prediction_pipeline"))

import numpy as np
import pandas as pd
from lstm_model import load_lstm_model, make_loader, get_predictions
from gru_model  import load_gru_model
from tcn_model  import load_tcn_model

FUSION_TICKERS = [
    "AAPL", "ADBE", "AMD", "AMZN", "CRM", "GOOGL",
    "INTC", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA",
]
DATE_START = "2025-01-23"
DATE_END   = "2026-02-26"

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data" / "processed" / "stock" / "dataset" / "relative_only_lb50"
MODEL_DIR  = REPO_ROOT / "outputs" / "price" / "models"


def main():
    # 1. Filter test_meta.csv to fusion-compatible rows
    meta = pd.read_csv(DATA_DIR / "test_meta.csv", parse_dates=["as_of_date"])
    mask = (
        meta["ticker"].isin(FUSION_TICKERS) &
        meta["as_of_date"].between(DATE_START, DATE_END)
    )
    idx = np.where(mask.values)[0]
    print(f"Total price test rows : {len(meta)}")
    print(f"Fusion-subset rows    : {len(idx)}")
    print(f"Date range            : {meta.loc[mask, 'as_of_date'].min().date()} "
          f"to {meta.loc[mask, 'as_of_date'].max().date()}")
    print(f"Tickers               : {sorted(meta.loc[mask, 'ticker'].unique())}\n")

    # 2. Load X / y for the filtered rows
    npz   = np.load(DATA_DIR / "test.npz")
    X_sub = npz["X"][idx]
    y_sub = npz["y"][idx]

    # 3. Inference for each model
    device = "cpu"
    models = [
        ("LSTM", load_lstm_model(MODEL_DIR / "lstm_best.pt", device)),
        ("GRU",  load_gru_model( MODEL_DIR / "gru_best.pt",  device)),
        ("TCN",  load_tcn_model( MODEL_DIR / "tcn_best.pt",  device)),
    ]

    rows = []
    for name, model in models:
        loader = make_loader(X_sub, y_sub, batch_size=512, shuffle=False)
        y_pred, y_true = get_predictions(model, loader, device)
        mae     = float(np.mean(np.abs(y_pred - y_true)))
        rmse    = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "DirAcc": dir_acc})
        print(f"{name:4s} | MAE={mae:.4f} | RMSE={rmse:.4f} | DirAcc={dir_acc:.4f}")

    print()
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
