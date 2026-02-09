import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

TEAMMATE_ID = "teammate6"
TEAMMATE_NAME = "Hanzhen Qin"
GDELT_START = datetime(2025, 6, 1)
GDELT_END = datetime(2025, 12, 31)

GDELT_THEMES = [
    "Federal Reserve interest rate technology",
    "Fed policy tech stocks",
    "technology regulation policy",
    "AI regulation artificial intelligence",
    "antitrust tech companies",
    "CHIPS act semiconductor",
    "chip shortage supply chain",
    "inflation technology sector",
    "recession tech stocks",
    "US China tech trade",
    "semiconductor geopolitics",
    "Silicon Valley",
]

def fetch_gdelt_news(keywords, start_date, end_date, max_records=250):
    """
    Fetch news articles from the GDELT Doc 2.1 API for a given keyword and time range.

    Args:
        keywords (str): Search query or theme keywords used for GDELT article retrieval.
        start_date (datetime.datetime): Start of the news collection period (inclusive).
        end_date (datetime.datetime): End of the news collection period (inclusive).
        max_records (int, optional): Maximum number of articles to retrieve in a single request. Must be <= 250 due to GDELT API limitations. Default is 250.

    Returns:
        List[dict]:
            A list of news article records. Each record is a dictionary containing:
            - datetime (str): Article timestamp provided by GDELT
            - headline (str): News title
            - url (str): Article URL
            - source (str): News source domain
            - language (str): Language code of the article
            - theme (str): Keyword/theme used for retrieval
            - data_source (str): Data source identifier ("gdelt")
            - collected_by (str): Teammate identifier
    """

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": keywords,
        "mode": "artlist",
        "maxrecords": min(max_records, 250),
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": end_date.strftime("%Y%m%d235959"),
        "format": "json",
        "sort": "datedesc",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        articles = []
        if ("articles" in data):
            for article in data["articles"]:
                articles.append({
                    "datetime": article.get("seendate"),
                    "headline": article.get("title"),
                    "url": article.get("url"),
                    "source": article.get("domain"),
                    "language": article.get("language"),
                    "theme": keywords,
                    "data_source": "gdelt",
                    "collected_by": TEAMMATE_ID,
                })
        return articles

    except requests.exceptions.Timeout:
        print("Request timed out, retrying...")
        time.sleep(3)
        return fetch_gdelt_news(keywords, start_date, end_date, max_records)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Data parsing failed: {e}")
        return []

def fetch_all_gdelt_news():
    """
    Collect GDELT news articles for all predefined themes within the specified
    teammate time period.

    This function iterates over all themes defined in GDELT_THEMES, retrieves
    news articles for each theme using the GDELT Doc API, and aggregates the
    results into a single list.

    Returns:
        List[dict]:
            A list of GDELT news article records collected across all themes.
            Each record follows the same schema as returned by fetch_gdelt_news().
    """
    print("\n" + "=" * 70)
    print("Starting GDELT Macro News Collection (Teammate 6)")
    print("=" * 70)
    print(f"Teammate: {TEAMMATE_NAME} ({TEAMMATE_ID})")
    print(f"Date range: {GDELT_START.date()} to {GDELT_END.date()}")
    print(f"Themes: {len(GDELT_THEMES)}")
    print("=" * 70)

    all_news = []
    for i, theme in enumerate(GDELT_THEMES, 1):
        print(f"\n[{i}/{len(GDELT_THEMES)}] {theme[:55]}...")

        news = fetch_gdelt_news(theme, GDELT_START, GDELT_END)
        all_news.extend(news)

        print(f"    Fetched {len(news):,} articles")

        if (i < len(GDELT_THEMES)):
            time.sleep(2)

    print(f"\nGDELT total (before dedup): {len(all_news):,} articles")
    return all_news

def save_outputs(df: pd.DataFrame):
    """
    Save collected news data to disk in JSON, CSV, and statistics report formats.

    The outputs are saved under the directory:
    data/raw/news/{TEAMMATE_ID}/, following the same structure and naming
    conventions as the leader's data collection scripts.

    Args:
        df (pandas.DataFrame):
            DataFrame containing collected news articles. Each row represents
            one news record with standardized fields (e.g., headline, url,
            datetime, data_source).

    Returns:
        pathlib.Path or None:
            Path to the saved JSON file if data is successfully written.
            None if the input DataFrame is empty.
    """
    output_dir = Path(f"data/raw/news/{TEAMMATE_ID}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if (df.empty):
        print("No data to save")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{TEAMMATE_ID}_news_{timestamp}"

    # JSON
    json_path = output_dir / f"{filename}.json"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    # CSV
    csv_path = output_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Stats
    stats_path = output_dir / f"{filename}_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("QianTick News Collection Statistics Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Collector: {TEAMMATE_NAME} ({TEAMMATE_ID})\n")
        f.write(f"Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("GDELT Tasks:\n")
        f.write(f"  Period: {GDELT_START.date()} to {GDELT_END.date()}\n")
        f.write(f"  Number of themes: {len(GDELT_THEMES)}\n\n")

        f.write("=" * 60 + "\n\n")
        f.write("Collection Results:\n")
        f.write(f"  Total: {len(df):,} articles\n\n")

        f.write("Data source distribution:\n")
        for source, count in df["data_source"].value_counts().items():
            f.write(f"  {source}: {count:,} articles\n")
        f.write("\n")

        if ("datetime" in df.columns and not df["datetime"].isna().all()):
            f.write(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}\n")

    file_size_mb = json_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print("Data saved successfully")
    print("=" * 70)
    print(f"  JSON:  {json_path} ({file_size_mb:.2f} MB)")
    print(f"  CSV:   {csv_path}")
    print(f"  Stats: {stats_path}")
    print("=" * 70)

    return json_path

def main():
    all_news = fetch_all_gdelt_news()
    df = pd.DataFrame(all_news)
    if (df.empty):
        print("\nNo data collected. Please check network.")
        return
    original = len(df)
    df = df.drop_duplicates(subset=["headline", "url"], keep="first")
    removed = original - len(df)
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    print(f"Total: {len(df):,} articles (removed {removed:,} duplicates)")
    print("=" * 70)
    saved_path = save_outputs(df)
    print("\nTask completed!")
    print(f"Saved JSON: {saved_path}")

if __name__ == "__main__":
    main()