import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRIMARY_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "AMD", "INTC", "CRM", "NFLX", "AVGO", "ORCL", "ADBE"]
MARKET_CONTEXT_TICKERS = ["SPY", "QQQ", "^VIX", "XLK", "SOXX", "TLT"]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stock"

def download_single_ticker(
    ticker: str, 
    start_date: str = DEFAULT_START_DATE, 
    end_date: str | None = None, 
    interval: str = "1d", 
    max_retries: int = 3
):
    if (end_date is None):
        end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {ticker}: ({start_date} -> {end_date}), attempt {attempt}/{max_retries}.")
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=False, progress=False)
            if (df is None or df.empty):
                print(f"WARNING: No data returned for {ticker}.")
                return None
            df = clean_price_dataframe(df, ticker)
            print(f"{ticker}: {len(df)} rows, [{df['Date'].iloc[0]} — {df['Date'].iloc[-1]}].")
            return df

        except Exception as exc:
            print(f"Attempt {attempt} failed for {ticker}: {exc}.")
            if (attempt < max_retries):
                time.sleep(2 * attempt)

    print(f"ERROR: All {max_retries} attempts failed for {ticker}.")
    return None

def clean_price_dataframe(df: pd.DataFrame, ticker: str):
    df = df.copy()
    # yfinance may return MultiIndex columns when downloading single ticker.
    if (isinstance(df.columns, pd.MultiIndex)):
        # Only keep the outer index
        df.columns = df.columns.get_level_values(0)
    if ("Date" not in df.columns):
        df = df.reset_index()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if (col not in df.columns):
            raise ValueError(f"Missing required column '{col}' for {ticker}.")
    # The adjusted closing price-also known as the ex-rights closing price-is the closing price 
    # that has been adjusted to account for factors such as stock splits and dividends.
    if ("Adj Close" not in df.columns):
        # If the certain stock doesn't have this column, keep its output as same as the "Close".
        df["Adj Close"] = df["Close"]
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    price_cols = ["Open", "High", "Low", "Close"]
    df = df.dropna(subset=price_cols, how="all").reset_index(drop=True)
    # If there is missing value in these columns, we will try to fill at most two consecutive NA 
    # value according to the closest line above. If there is missing more than three lines, we can only abandon these data.
    df[price_cols + ["Adj Close", "Volume"]] = ( df[price_cols + ["Adj Close", "Volume"]].ffill(limit=2))
    df = df.dropna(subset=price_cols).reset_index(drop=True)

    for col in price_cols + ["Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["Volume"] = df["Volume"].fillna(0).astype("int64")
    col_order = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[col_order]

def download_all_tickers(
    tickers: list[str], 
    start_date: str = DEFAULT_START_DATE, 
    end_date: str | None = None, 
    interval: str = "1d", 
    pause: float = 0.5
):
    results = {}
    for i, ticker in enumerate(tickers):
        df = download_single_ticker(ticker, start_date, end_date, interval)
        if (df is not None):
            results[ticker] = df
        if (i < len(tickers) - 1):
            time.sleep(pause)
    print(f"Downloaded {len(results)} / {len(tickers)} tickers successfully.")
    return results

def save_raw_data(data_dict: dict[str, pd.DataFrame], out_dir: str | Path, save_merged: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, df in data_dict.items():
        path = out_dir / f"{ticker}.csv"
        df.to_csv(path, index=False)
    if (save_merged and data_dict):
        merged = pd.concat(data_dict.values(), ignore_index=True)
        merged = merged.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        merged_path = out_dir / "all_stocks.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Saved merged CSV to {merged_path} ({len(merged)} rows, {merged['Ticker'].nunique()} tickers).")
    return out_dir

def fetch_all_stocks(
    primary_tickers: list[str] = PRIMARY_TICKERS, 
    market_tickers: list[str] = MARKET_CONTEXT_TICKERS, 
    start_date: str = DEFAULT_START_DATE, 
    end_date: str | None = None, 
    out_dir: str | Path = DEFAULT_RAW_DIR, 
    include_market_context: bool = True
):
    all_tickers = list(primary_tickers)
    if (include_market_context):
        all_tickers += [t for t in market_tickers if (t not in all_tickers)]
    data = download_all_tickers(all_tickers, start_date, end_date)
    if (not data):
        raise RuntimeError("No tickers were downloaded successfully.")
    save_raw_data(data, out_dir)
    total_rows = sum(len(df) for df in data.values())
    date_min = min(df["Date"].min() for df in data.values())
    date_max = max(df["Date"].max() for df in data.values())
    print(f"fetch_data complete: {len(data)} tickers, {total_rows} total rows, date range"
          f" [{date_min.strftime('%Y-%m-%d')} — {date_max.strftime('%Y-%m-%d')}].")
    return data