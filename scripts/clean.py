import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

input_path = ROOT / "data/processed/news/finnhub_merged.csv"
df = pd.read_csv(input_path)

cleaned = pd.DataFrame({
    "ticker": df["symbol"].astype(str),
    "date": pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d"),
    "article_id": df["url"].astype(str),
    "headline": df["headline"].fillna("").astype(str),
})

cleaned = cleaned.drop_duplicates(subset=["article_id"])

output_path = ROOT / "data/processed/news/news_cleaned.parquet"

output_path.parent.mkdir(parents=True, exist_ok=True)

cleaned.to_parquet(output_path, index=False)

print("Saved:", cleaned.shape)
print("Path:", output_path)