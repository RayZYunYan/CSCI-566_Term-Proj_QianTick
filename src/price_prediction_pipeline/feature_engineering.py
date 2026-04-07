from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PIPELINE_ROOT / "data"
REQUIRED_RAW_COLS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]

# Keep the original basic feature names for continuity with your current pipeline.
BASIC_FEATURE_COLS = [
    "daily_return", 
    "log_return", 
    "high_low_range", 
    "open_close_change", 
    "volume_change", 
    "close_rolling_mean_20", 
    "close_rolling_std_20"
]

# Sequence models should consume relatively stationary / scale-friendly inputs
SEQUENCE_FEATURE_COLS = [
    "open_to_prev_close", 
    "high_to_prev_close", 
    "low_to_prev_close", 
    "close_to_prev_close", 
    "log_volume", 
    "daily_return", 
    "log_return", 
    "high_low_range", 
    "open_close_change", 
    "volume_change", 
    "close_rolling_mean_20", 
    "close_rolling_std_20"
]

# Technical indicators: keep the classic indicators, but make them more useful for modeling by also including several relative/state-style features.
TECH_INDICATOR_COLS = [
    # classic moving averages
    "sma_5", "sma_10", "sma_20", 
    "ema_5", "ema_10", "ema_20", 
    # relative price-vs-trend features
    "close_vs_sma_5", "close_vs_sma_10", "close_vs_sma_20", 
    "close_vs_ema_5", "close_vs_ema_10", "close_vs_ema_20", 
    # momentum / oscillator
    "rsi_14", 
    "macd", "macd_signal", "macd_hist", 
    "momentum_10", "momentum_10_pct", 
    # Bollinger
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_percent_b", 
    # volatility
    "atr_14", "atr_ratio", 
    # volume
    "volume_sma_20", "volume_ratio"
]

FEATURE_PRESET_REGISTRY = {
    # Keep the current full feature set as the default baseline.
    "full": list(dict.fromkeys(SEQUENCE_FEATURE_COLS + TECH_INDICATOR_COLS)), 
    # Prefer relative / scale-free indicators for pooled multi-ticker training.
    "relative_only": list(dict.fromkeys(SEQUENCE_FEATURE_COLS + [
        "close_vs_sma_5", "close_vs_sma_10", "close_vs_sma_20", 
        "close_vs_ema_5", "close_vs_ema_10", "close_vs_ema_20", 
        "rsi_14", 
        "macd", "macd_signal", "macd_hist", 
        "momentum_10_pct", 
        "bb_width", "bb_percent_b", 
        "atr_ratio", 
        "volume_ratio"
    ]))
}

def get_sequence_feature_columns():
    return list(SEQUENCE_FEATURE_COLS)

def get_basic_feature_columns():
    return list(BASIC_FEATURE_COLS)

def get_technical_indicator_columns():
    return list(TECH_INDICATOR_COLS)

def get_feature_preset_columns(feature_preset: str = "full"):
    if (feature_preset not in FEATURE_PRESET_REGISTRY):
        valid = sorted(FEATURE_PRESET_REGISTRY.keys())
        raise ValueError(f"Unknown feature_preset '{feature_preset}'. Valid options: {valid}")
    return list(FEATURE_PRESET_REGISTRY[feature_preset])

def safe_divide(numerator: pd.Series, denominator: pd.Series | float | int):
    if (isinstance(denominator, pd.Series)):
        denominator = denominator.replace(0, np.nan)
    elif (denominator == 0):
        denominator = np.nan
    return numerator / denominator

def validate_input_columns(df: pd.DataFrame, required_cols: Iterable[str] = REQUIRED_RAW_COLS):
    missing = [c for c in required_cols if (c not in df.columns)]
    if (missing):
        raise ValueError(f"Missing required columns: {missing}")

def ensure_sorted_single_ticker(df: pd.DataFrame):
    out = df.sort_values("Date").reset_index(drop=True).copy()
    # Check the number of the ticker in the dataframe.
    if ("Ticker" in out.columns and out["Ticker"].nunique() > 1):
        raise ValueError("engineer_features_single_ticker expects data from exactly one ticker.")
    return out

def replace_inf_with_nan(df: pd.DataFrame):
    return df.replace([np.inf, -np.inf], np.nan)

def compute_basic_features(df: pd.DataFrame):
    df = df.copy()
    # Get the previous day's closing price for each row.
    prev_close = df["Close"].shift(1)
    
    # Measure how today's open price changes relative to the previous close.
    df["open_to_prev_close"] = safe_divide(df["Open"] - prev_close, prev_close)
    
    # Measure how today's high price changes relative to the previous close.
    df["high_to_prev_close"] = safe_divide(df["High"] - prev_close, prev_close)
    
    # Measure how today's low price changes relative to the previous close.
    df["low_to_prev_close"] = safe_divide(df["Low"] - prev_close, prev_close)
    
    # Measure how today's close price changes relative to the previous close.
    df["close_to_prev_close"] = safe_divide(df["Close"] - prev_close, prev_close)
    
    # Use log-transformed volume to make trading volume more stable and less skewed.
    df["log_volume"] = np.log1p(df["Volume"])
    
    # Compute the simple daily return based on closing prices.
    df["daily_return"] = df["Close"].pct_change()
    
    # Compute the log return, which is often useful in financial time series.
    df["log_return"] = np.log(safe_divide(df["Close"], prev_close))
    
    # Measure the intraday price range relative to the closing price.
    df["high_low_range"] = safe_divide(df["High"] - df["Low"], df["Close"])
    
    # Measure the price change from market open to market close.
    df["open_close_change"] = safe_divide(df["Close"] - df["Open"], df["Open"])
    
    # Compute the percentage change in trading volume from the previous day.
    df["volume_change"] = df["Volume"].pct_change()
    
    # Clip extreme volume spikes so unusually large changes do not dominate the feature values.
    df["volume_change"] = df["volume_change"].clip(-10, 10)
    
    # Compute the 20-day rolling mean of closing prices to capture the local trend.
    rolling_mean_20 = df["Close"].rolling(window=20, min_periods=20).mean()
    
    # Compute the 20-day rolling standard deviation of closing prices to capture recent volatility.
    rolling_std_20 = df["Close"].rolling(window=20, min_periods=20).std()
    
    # Measure how far the current close is from its 20-day average in relative terms.
    df["close_rolling_mean_20"] = safe_divide(df["Close"] - rolling_mean_20, rolling_mean_20)
    
    # Measure recent volatility relative to the 20-day average price level.
    df["close_rolling_std_20"] = safe_divide(rolling_std_20, rolling_mean_20)
    return df

def rsi_wilder(series: pd.Series, period: int = 14):
    # Compute the day-to-day price difference.
    delta = series.diff()
    # Keep only positive price changes as gains, and set negative changes to 0.
    gain = delta.clip(lower=0)
    # Keep only negative price changes as losses, and convert them to positive values.
    loss = -delta.clip(upper=0)
    # Compute Wilder's smoothed average gain using exponential weighted mean.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Compute Wilder's smoothed average loss using exponential weighted mean.
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Compute the relative strength as the ratio of average gain to average loss.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    # Convert relative strength into RSI values on a 0 to 100 scale.
    rsi = 100 - (100 / (1 + rs))
    # If there is no average loss but there is average gain, set RSI to 100.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # If both average gain and average loss are 0, set RSI to the neutral value 50.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    # Compute the fast exponential moving average of the price series.
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    # Compute the slow exponential moving average of the price series.
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    # Compute the MACD line as the difference between fast EMA and slow EMA.
    macd_line = ema_fast - ema_slow
    # Compute the signal line as the exponential moving average of the MACD line.
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    # Compute the MACD histogram as the difference between MACD line and signal line.
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr_wilder(df: pd.DataFrame, period: int = 14):
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - prev_close).abs()
    low_prev_close = (df["Low"] - prev_close).abs()
    # Combine the three candidate range measures and take the largest value as the true range for each row.
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    # Apply Wilder-style exponential smoothing to the true range to get the ATR series.
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Return the final average true range value.
    return atr

def compute_technical_indicators(df: pd.DataFrame):
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]
    
    # Moving averages
    for w in [5, 10, 20]:
        sma = close.rolling(window=w, min_periods=w).mean()
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()

        df[f"sma_{w}"] = sma
        df[f"ema_{w}"] = ema

        df[f"close_vs_sma_{w}"] = safe_divide(close - sma, sma)
        df[f"close_vs_ema_{w}"] = safe_divide(close - ema, ema)
    
    # RSI
    df["rsi_14"] = rsi_wilder(close, period=14)
    
    # MACD
    macd_line, signal_line, hist = macd(close)
    # Normalize the MACD line by the current close price to make it scale-free across different price levels.
    df["macd"] = safe_divide(macd_line, close)
    # Normalize the MACD signal line by the current close price for better comparability and model stability.
    df["macd_signal"] = safe_divide(signal_line, close)
    # Normalize the MACD histogram by the current close price to represent relative momentum strength.
    df["macd_hist"] = safe_divide(hist, close)
    
    # Bollinger Bands
    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_band_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = safe_divide(bb_upper - bb_lower, bb_mid)
    df["bb_percent_b"] = safe_divide(close - bb_lower, bb_band_range)
    
    # ATR
    df["atr_14"] = atr_wilder(df, period=14)
    df["atr_ratio"] = safe_divide(df["atr_14"], close)
    
    # Momentum
    df["momentum_10"] = close - close.shift(10)
    df["momentum_10_pct"] = safe_divide(close - close.shift(10), close.shift(10))
    
    # Volume features
    volume_sma_20 = volume.rolling(window=20, min_periods=20).mean()
    df["volume_sma_20"] = volume_sma_20
    df["volume_ratio"] = safe_divide(volume, volume_sma_20)
    return df

def finalize_feature_table( df: pd.DataFrame, required_feature_cols: list[str] | None = None):
    df = replace_inf_with_nan(df)
    if (required_feature_cols is None):
        required_feature_cols = get_feature_preset_columns("full")
    cols_in_df = [c for c in required_feature_cols if (c in df.columns)]
    df = df.dropna(subset=cols_in_df).reset_index(drop=True)
    return df

def engineer_features_single_ticker(ticker_df: pd.DataFrame):
    validate_input_columns(ticker_df)
    ticker_df = ensure_sorted_single_ticker(ticker_df)
    ticker_df = compute_basic_features(ticker_df)
    ticker_df = compute_technical_indicators(ticker_df)
    ticker_df = finalize_feature_table(ticker_df)
    ticker_df = replace_inf_with_nan(ticker_df).fillna(0.0)
    return ticker_df

def engineer_features_all(input_path: str | Path | None = None, output_dir: str | Path | None = None):
    if (input_path is None):
        input_path = DATA_ROOT / "raw" / "stock" / "all_stocks.csv"
    if (output_dir is None):
        output_dir = DATA_ROOT / "processed" / "stock" / "features"
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path, parse_dates=["Date"])
    validate_input_columns(raw)
    all_results = []
    tickers = sorted(raw["Ticker"].dropna().unique())

    for ticker in tickers:
        ticker_df = raw.loc[raw["Ticker"] == ticker].copy()
        featured = engineer_features_single_ticker(ticker_df)
        all_results.append(featured)

    if (not all_results):
        raise ValueError("No ticker data was processed. Check the input file.")
    final_merged_feature = pd.concat(all_results, ignore_index=True)
    final_merged_feature = final_merged_feature.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    merged_path = output_dir / "all_features.csv"
    final_merged_feature.to_csv(merged_path, index=False)