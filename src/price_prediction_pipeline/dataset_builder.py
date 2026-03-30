from pathlib import Path
import numpy as np
import pandas as pd
from feature_engineering import get_sequence_feature_columns, get_technical_indicator_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def build_windows_for_ticker(ticker_df: pd.DataFrame, lookback: int, seq_feature_cols: list[str], tech_feature_cols: list[str]):
    # Sort rows by date in ascending order and reset the row index.
    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    # Need at least lookback days for input and 1 additional day for the label.
    if (len(ticker_df) < lookback + 1):
        return [], [], [], []
    # Get the ticker symbol for this single-ticker DataFrame.
    ticker = ticker_df["Ticker"].iloc[0]
    # Extract dates for metadata.
    dates = ticker_df["Date"].values
    # Extract close prices for next-day up/down label construction.
    close = ticker_df["Close"].values
    # Extract sequential features as a 2D NumPy array of shape (T, n_seq_features).
    seq_data = ticker_df[seq_feature_cols].values
    # Extract technical indicator features as a 2D NumPy array of shape (T, n_tech_features).
    tech_data = ticker_df[tech_feature_cols].values
    # Initialize containers for sequence inputs, technical inputs, labels, and metadata.
    X_seq_list, X_tech_list, y_list, meta_list = [], [], [], []
    # Slide a lookback window over time and build one sample at each valid time step t.
    for t in range(lookback - 1, len(ticker_df) - 1):
        # Compute the starting index of the current lookback window.
        window_start = t - lookback + 1
        # Use the past (lookback) days of sequential features up to and including day t.
        X_seq_list.append(seq_data[window_start : t + 1])
        # Use only the technical indicator snapshot at the current day t.
        X_tech_list.append(tech_data[t])
        # Create a binary label: 1 if next day's close is higher than today's, else 0.
        label = 1 if (close[t + 1] > close[t]) else 0
        y_list.append(label)

        # Store metadata for tracing the sample's input date and prediction target date.
        meta_list.append({
            "ticker": ticker, 
            "as_of_date": dates[t], 
            "target_date": dates[t + 1]
        })
    return X_seq_list, X_tech_list, y_list, meta_list

def combine_all_tickers(feature_df: pd.DataFrame, lookback: int, seq_feature_cols: list[str] | None = None, tech_feature_cols: list[str] | None = None):
    if (seq_feature_cols is None):
        seq_feature_cols = get_sequence_feature_columns()
    if (tech_feature_cols is None):
        tech_feature_cols = get_technical_indicator_columns()
    all_X_seq, all_X_tech, all_y, all_meta = [], [], [], []
    # Collect all types of tickers
    tickers = sorted(feature_df["Ticker"].unique())
    for ticker in tickers:
        tdf = feature_df[feature_df["Ticker"] == ticker].copy()
        xs, xt, yl, ml = build_windows_for_ticker(tdf, lookback, seq_feature_cols, tech_feature_cols)
        all_X_seq.extend(xs)
        all_X_tech.extend(xt)
        all_y.extend(yl)
        all_meta.extend(ml)
    
    X_seq = np.array(all_X_seq, dtype=np.float32)
    X_tech = np.array(all_X_tech, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    meta_df = pd.DataFrame(all_meta)
    meta_df["as_of_date"] = pd.to_datetime(meta_df["as_of_date"])
    meta_df["target_date"] = pd.to_datetime(meta_df["target_date"])
    # Sort the entire dataset by as_of_date to ensure chronological order for time-based splitting.
    sort_idx = meta_df["as_of_date"].argsort().values
    X_seq = X_seq[sort_idx]
    X_tech = X_tech[sort_idx]
    y = y[sort_idx]
    meta_df = meta_df.iloc[sort_idx].reset_index(drop=True)
    return X_seq, X_tech, y, meta_df

def time_based_split(
    X_seq: np.ndarray, 
    X_tech: np.ndarray, 
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
            "X_seq": X_seq[start : end], 
            "X_tech": X_tech[start : end], 
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
    seq_mean, seq_std = fit_array_normalization(splits["train"]["X_seq"])
    tech_mean, tech_std = fit_array_normalization(splits["train"]["X_tech"])
    norm_stats = {"seq_mean": seq_mean, "seq_std": seq_std, "tech_mean": tech_mean, "tech_std": tech_std}
    for name in ["train", "val", "test"]:
        splits[name]["X_seq"] = apply_array_normalization(splits[name]["X_seq"], seq_mean, seq_std)
        splits[name]["X_tech"] = apply_array_normalization(splits[name]["X_tech"], tech_mean, tech_std)
    # replace any NaN/inf that survived
    for name in ["train", "val", "test"]:
        splits[name]["X_seq"] = np.nan_to_num(splits[name]["X_seq"], nan=0.0, posinf=0.0, neginf=0.0)
        splits[name]["X_tech"] = np.nan_to_num(splits[name]["X_tech"], nan=0.0, posinf=0.0, neginf=0.0)
    return splits, norm_stats

def save_dataset(splits: dict, norm_stats: dict, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        np.savez_compressed(
            out_dir / f"{name}.npz", 
            X_seq=splits[name]["X_seq"], 
            X_tech=splits[name]["X_tech"], 
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
            "X_seq": npz["X_seq"], 
            "X_tech": npz["X_tech"], 
            "y": npz["y"], 
            "meta": pd.read_csv(data_dir / f"{name}_meta.csv", parse_dates=["as_of_date", "target_date"])
        }
    norm_stats = dict(np.load(data_dir / "norm_stats.npz"))
    return splits, norm_stats

def build_dataset(
    feature_path: str | Path | None = None, 
    output_dir: str | Path | None = None, 
    lookback: int = 30, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15
):
    if (feature_path is None):
        feature_path = PROJECT_ROOT / "data" / "processed" / "stock" / "features" / "all_features.csv"
    if (output_dir is None):
        output_dir = PROJECT_ROOT / "data" / "processed" / "stock" / "dataset"
    feature_path = Path(feature_path)
    output_dir = Path(output_dir)
    feature_df = pd.read_csv(feature_path, parse_dates=["Date"])

    # Build windowed samples.
    X_seq, X_tech, y, meta_df = combine_all_tickers(feature_df, lookback)

    # Time-based split.
    splits = time_based_split(X_seq, X_tech, y, meta_df, train_ratio, val_ratio)

    # Normalize using train-set statistics only.
    splits, norm_stats = normalize_splits(splits)

    # Save to disk.
    save_dataset(splits, norm_stats, output_dir)