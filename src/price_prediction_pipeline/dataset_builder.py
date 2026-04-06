from pathlib import Path
import numpy as np
import pandas as pd
from feature_engineering import get_feature_preset_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def build_windows_for_ticker(ticker_df: pd.DataFrame, lookback: int, feature_cols: list[str]):
    # Sort rows by date in ascending order and reset the row index.
    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    # Need at least lookback days for input and 1 additional day for the label.
    if (len(ticker_df) < lookback + 1):
        return [], [], []
    # Get the ticker symbol for this single-ticker DataFrame.
    ticker = ticker_df["Ticker"].iloc[0]
    # Extract dates for metadata.
    dates = ticker_df["Date"].values
    # Extract close prices for next-day up/down label construction.
    close = ticker_df["Close"].values
    # Extract all features as a 2D NumPy array of shape (T, num_features).
    feature_data = ticker_df[feature_cols].values
    # Initialize containers for feature windows, labels, and metadata.
    X_list, y_list, meta_list = [], [], []
    # Slide a lookback window over time and build one sample at each valid time step t.
    for t in range(lookback - 1, len(ticker_df) - 1):
        # Compute the starting index of the current lookback window.
        window_start = t - lookback + 1
        # Use the past (lookback) days of all features up to and including day t.
        X_list.append(feature_data[window_start : t + 1])
        # Create a binary label: 1 if next day's close is higher than today's, else 0.
        label = 1 if (close[t + 1] > close[t]) else 0
        y_list.append(label)
        # Store metadata for tracing the sample's input date and prediction target date.
        meta_list.append({
            "ticker": ticker, 
            "as_of_date": dates[t], 
            "target_date": dates[t + 1]
        })
    return X_list, y_list, meta_list

def combine_all_tickers(feature_df: pd.DataFrame, lookback: int, feature_cols: list[str] | None = None):
    if (feature_cols is None):
        feature_cols = get_feature_preset_columns("full")
    all_X, all_y, all_meta = [], [], []
    # Collect samples from all tickers.
    tickers = sorted(feature_df["Ticker"].unique())
    for ticker in tickers:
        tdf = feature_df[feature_df["Ticker"] == ticker].copy()
        xl, yl, ml = build_windows_for_ticker(tdf, lookback, feature_cols)
        all_X.extend(xl)
        all_y.extend(yl)
        all_meta.extend(ml)
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    meta_df = pd.DataFrame(all_meta)
    meta_df["as_of_date"] = pd.to_datetime(meta_df["as_of_date"])
    meta_df["target_date"] = pd.to_datetime(meta_df["target_date"])
    # Sort the entire dataset by as_of_date to ensure chronological order for time-based splitting.
    sort_idx = meta_df["as_of_date"].argsort().values
    X = X[sort_idx]
    y = y[sort_idx]
    meta_df = meta_df.iloc[sort_idx].reset_index(drop=True)
    return X, y, meta_df

def time_based_split(
    X: np.ndarray, 
    y: np.ndarray, 
    meta_df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15
):
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = {}
    # Split the dataset into train/val/test based on chronological order of as_of_date.
    for name, start, end in [("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, n)]:
        splits[name] = {
            "X": X[start : end], 
            "y": y[start : end], 
            "meta": meta_df.iloc[start : end].reset_index(drop=True)
        }
    return splits

def fit_array_normalization(X: np.ndarray):
    if (X.ndim == 3):
        # Reshape from (N, T, F) to (N*T, F) to compute mean/std across all time steps and samples for each feature.
        flat = X.reshape(-1, X.shape[-1])
    else:
        flat = X
    # Compute mean and std, ignoring NaNs. Convert to float32 for consistency and memory efficiency.
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    # Avoid division by zero.
    std[std == 0] = 1.0
    return mean, std

def apply_array_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)

def normalize_splits(splits: dict):
    # Compute normalization statistics exclusively from the training set to prevent data leakage.
    mean, std = fit_array_normalization(splits["train"]["X"])
    norm_stats = {"mean": mean, "std": std}
    # Apply train-set normalization to all splits (train, val, test).
    for name in ["train", "val", "test"]:
        splits[name]["X"] = apply_array_normalization(splits[name]["X"], mean, std)
    # Replace any NaN/inf that survived normalization with zero.
    for name in ["train", "val", "test"]:
        splits[name]["X"] = np.nan_to_num(splits[name]["X"], nan=0.0, posinf=0.0, neginf=0.0)
    return splits, norm_stats

def save_dataset(splits: dict, norm_stats: dict, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        np.savez_compressed(
            out_dir / f"{name}.npz", 
            X=splits[name]["X"], 
            y=splits[name]["y"]
        )
        splits[name]["meta"].to_csv(out_dir / f"{name}_meta.csv", index=False)
    np.savez(out_dir / "norm_stats.npz", **norm_stats)

def load_dataset(data_dir: str | Path):
    data_dir = Path(data_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        npz = np.load(data_dir / f"{name}.npz")
        splits[name] = {
            "X": npz["X"], 
            "y": npz["y"], 
            "meta": pd.read_csv(data_dir / f"{name}_meta.csv", parse_dates=["as_of_date", "target_date"])
        }
    norm_stats = dict(np.load(data_dir / "norm_stats.npz"))
    return splits, norm_stats

def build_dataset(
    feature_path: str | Path | None = None, 
    output_dir: str | Path | None = None, 
    feature_preset: str = "full", 
    lookback: int = 30, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15
):
    if (feature_path is None):
        feature_path = PROJECT_ROOT / "data" / "processed" / "stock" / "features" / "all_features.csv"
    if (output_dir is None):
        output_dir = PROJECT_ROOT / "data" / "processed" / "stock" / "dataset" / f"{feature_preset}_lb{lookback}"
    feature_path = Path(feature_path)
    output_dir = Path(output_dir)
    feature_df = pd.read_csv(feature_path, parse_dates=["Date"])
    feature_cols = get_feature_preset_columns(feature_preset)

    # Build windowed samples using the specified feature preset.
    X, y, meta_df = combine_all_tickers(feature_df, lookback, feature_cols)

    # Time-based split.
    splits = time_based_split(X, y, meta_df, train_ratio, val_ratio)

    # Normalize using train-set statistics only.
    splits, norm_stats = normalize_splits(splits)

    # Save to disk.
    save_dataset(splits, norm_stats, output_dir)