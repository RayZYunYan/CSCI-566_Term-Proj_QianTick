"""
QianTick Team News Collector
================================================
Data Sources: Finnhub (Company News) + GDELT (Macro News)
Time Range: 2022-06-01 to 2025-12-31
Stock Count: 15 Tech Stocks
Team Size: 6 Members
================================================
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ============================================
# Team Configuration
# ============================================

# 15 Selected Tech Stocks
ALL_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL',   # Tech Giants
    'NVDA', 'QCOM', 'INTC',     # Semiconductors
    'AMZN', 'META', 'AMD',      # Cloud/Social
    'TSLA', 'NFLX', 'CRM',      # Emerging Tech
    'ORCL', 'ADBE', 'SHOP',     # Enterprise Software/E-commerce
]

# Data Time Range
DATA_START_DATE = datetime(2022, 6, 1)
DATA_END_DATE = datetime(2025, 12, 31)

# Team Task Assignment
TEAM_CONFIG = {
    'teammate1': {
        'name': 'Ray',  # Your name
        'stocks': ['AAPL', 'MSFT', 'GOOGL'],
        'gdelt_start': datetime(2022, 6, 1),
        'gdelt_end': datetime(2023, 1, 31),
        'finnhub': True,
        'gdelt': True,
    },
    'teammate2': {
        'name': 'Teammate 2',
        'stocks': ['NVDA', 'QCOM', 'INTC'],
        'gdelt_start': datetime(2023, 2, 1),
        'gdelt_end': datetime(2023, 8, 31),
        'finnhub': True,
        'gdelt': True,
    },
    'teammate3': {
        'name': 'Teammate 3',
        'stocks': ['AMZN', 'META', 'AMD'],
        'gdelt_start': datetime(2023, 9, 1),
        'gdelt_end': datetime(2024, 3, 31),
        'finnhub': True,
        'gdelt': True,
    },
    'teammate4': {
        'name': 'Teammate 4',
        'stocks': ['TSLA', 'NFLX', 'CRM'],
        'gdelt_start': datetime(2024, 4, 1),
        'gdelt_end': datetime(2024, 10, 31),
        'finnhub': True,
        'gdelt': True,
    },
    'teammate5': {
        'name': 'Teammate 5',
        'stocks': ['ORCL', 'ADBE', 'SHOP'],
        'gdelt_start': datetime(2024, 11, 1),
        'gdelt_end': datetime(2025, 5, 31),
        'finnhub': True,
        'gdelt': True,
    },
    'teammate6': {
        'name': 'Teammate 6',
        'stocks': [],  # Not responsible for Finnhub collection
        'gdelt_start': datetime(2025, 6, 1),
        'gdelt_end': datetime(2025, 12, 31),
        'finnhub': False,
        'gdelt': True,
    },
}

# GDELT Macro News Themes
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


# ============================================
# News Collector
# ============================================

class TeamNewsCollector:
    """Team Collaborative News Collector"""

    def __init__(self, teammate_id):
        """
        Initialize the collector

        Args:
            teammate_id: Teammate ID, e.g. 'teammate1', 'teammate2', ...
        """
        if teammate_id not in TEAM_CONFIG:
            raise ValueError(
                f"Invalid teammate_id: {teammate_id}\n"
                f"Options: {list(TEAM_CONFIG.keys())}"
            )

        self.teammate_id = teammate_id
        self.config = TEAM_CONFIG[teammate_id]

        # API Key (only Finnhub needed)
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        if self.config['finnhub'] and not self.finnhub_key:
            raise ValueError(
                "FINNHUB_API_KEY not found!\n"
                "Please set it in the .env file: FINNHUB_API_KEY=your_key"
            )

        # Output directory
        self.output_dir = Path(f"data/raw/news/{teammate_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._print_config()


    def _print_config(self):
        """Print configuration info"""
        print("\n" + "=" * 70)
        print(f"Teammate: {self.config['name']} ({self.teammate_id})")
        print("=" * 70)

        if self.config['finnhub']:
            print(f"Finnhub Tasks:")
            print(f"   Stocks: {', '.join(self.config['stocks'])}")
            print(f"   Period: {DATA_START_DATE.date()} to {DATA_END_DATE.date()}")
            print(f"   Estimated: ~10,000-12,000 articles")

        if self.config['gdelt']:
            gdelt_months = (self.config['gdelt_end'].year - self.config['gdelt_start'].year) * 12
            gdelt_months += self.config['gdelt_end'].month - self.config['gdelt_start'].month + 1

            print(f"\nGDELT Tasks:")
            print(f"   Period: {self.config['gdelt_start'].date()} to {self.config['gdelt_end'].date()}")
            print(f"   Duration: {gdelt_months} months")
            print(f"   Themes: {len(GDELT_THEMES)}")
            print(f"   Estimated: ~{gdelt_months * len(GDELT_THEMES) * 150:,} articles")

        print("=" * 70 + "\n")


    # ========================================
    # Finnhub Company News
    # ========================================

    def fetch_finnhub_news(self, symbol, start_date, end_date):
        """
        Fetch news for a single stock from Finnhub

        Args:
            symbol: Stock ticker symbol
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'

        Returns:
            list: List of news articles
        """
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': start_date,
            'to': end_date,
            'token': self.finnhub_key
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data:
                articles.append({
                    'symbol': symbol,
                    'datetime': item.get('datetime'),
                    'headline': item.get('headline'),
                    'summary': item.get('summary'),
                    'source': item.get('source'),
                    'url': item.get('url'),
                    'image': item.get('image'),
                    'data_source': 'finnhub',
                    'collected_by': self.teammate_id
                })

            return articles

        except requests.exceptions.Timeout:
            print(f"      Request timed out, retrying...")
            time.sleep(3)
            return self.fetch_finnhub_news(symbol, start_date, end_date)

        except requests.exceptions.RequestException as e:
            print(f"      Request failed: {e}")
            return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"      Data parsing failed: {e}")
            return []


    def fetch_all_finnhub_news(self):
        """Fetch Finnhub news for all assigned stocks"""
        if not self.config['finnhub']:
            print("This teammate is not responsible for Finnhub collection")
            return []

        print("\n" + "=" * 70)
        print("Starting Finnhub Company News Collection")
        print("=" * 70)

        all_news = []
        stocks = self.config['stocks']

        start_str = DATA_START_DATE.strftime('%Y-%m-%d')
        end_str = DATA_END_DATE.strftime('%Y-%m-%d')

        for i, symbol in enumerate(stocks, 1):
            print(f"\n[{i}/{len(stocks)}] {symbol}")
            print(f"    Date range: {start_str} to {end_str}")

            news = self.fetch_finnhub_news(symbol, start_str, end_str)
            all_news.extend(news)

            print(f"    Fetched {len(news):,} articles")

            # API limit: 60 calls/min, 1 per second is sufficient
            if i < len(stocks):  # No need to wait after the last stock
                time.sleep(1.5)

        print(f"\nFinnhub total: {len(all_news):,} articles")
        return all_news


    # ========================================
    # GDELT Macro News
    # ========================================

    def fetch_gdelt_news(self, keywords, start_date, end_date, max_records=250):
        """
        Fetch news from GDELT

        Args:
            keywords: Search keywords
            start_date: Start date (datetime object)
            end_date: End date (datetime object)
            max_records: Maximum number of records (GDELT limit is 250)

        Returns:
            list: List of news articles
        """
        url = "https://api.gdeltproject.org/api/v2/doc/doc"

        params = {
            'query': keywords,
            'mode': 'artlist',
            'maxrecords': min(max_records, 250),
            'startdatetime': start_date.strftime('%Y%m%d000000'),
            'enddatetime': end_date.strftime('%Y%m%d235959'),
            'format': 'json',
            'sort': 'datedesc'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = []
            if 'articles' in data:
                for article in data['articles']:
                    articles.append({
                        'datetime': article.get('seendate'),
                        'headline': article.get('title'),
                        'url': article.get('url'),
                        'source': article.get('domain'),
                        'language': article.get('language'),
                        'theme': keywords,
                        'data_source': 'gdelt',
                        'collected_by': self.teammate_id
                    })

            return articles

        except requests.exceptions.Timeout:
            print(f"      Request timed out, retrying...")
            time.sleep(3)
            return self.fetch_gdelt_news(keywords, start_date, end_date, max_records)

        except requests.exceptions.RequestException as e:
            print(f"      Request failed: {e}")
            return []
        except (json.JSONDecodeError, TypeError) as e:
            print(f"      Data parsing failed: {e}")
            return []


    def fetch_all_gdelt_news(self):
        """Fetch all GDELT news for the assigned time period"""
        if not self.config['gdelt']:
            print("This teammate is not responsible for GDELT collection")
            return []

        print("\n" + "=" * 70)
        print("Starting GDELT Macro News Collection")
        print("=" * 70)

        gdelt_start = self.config['gdelt_start']
        gdelt_end = self.config['gdelt_end']

        print(f"Date range: {gdelt_start.date()} to {gdelt_end.date()}")

        all_news = []

        for i, theme in enumerate(GDELT_THEMES, 1):
            print(f"\n[{i}/{len(GDELT_THEMES)}] {theme[:55]}...")

            news = self.fetch_gdelt_news(theme, gdelt_start, gdelt_end)
            all_news.extend(news)

            print(f"    Fetched {len(news):,} articles")

            # Avoid sending requests too quickly
            if i < len(GDELT_THEMES):
                time.sleep(2)

        print(f"\nGDELT total: {len(all_news):,} articles")
        return all_news


    # ========================================
    # Collection and Saving
    # ========================================

    def collect_all(self):
        """
        Collect all assigned news data

        Returns:
            DataFrame: Collected news data
        """
        print("\n" + "=" * 70)
        print(f"Starting collection task for {self.config['name']}")
        print("=" * 70)

        all_news = []

        # 1. Finnhub Company News
        if self.config['finnhub']:
            finnhub_news = self.fetch_all_finnhub_news()
            all_news.extend(finnhub_news)

        # 2. GDELT Macro News
        if self.config['gdelt']:
            gdelt_news = self.fetch_all_gdelt_news()
            all_news.extend(gdelt_news)

        # Convert to DataFrame
        df = pd.DataFrame(all_news)

        if not df.empty:
            # Deduplicate
            original = len(df)
            df = df.drop_duplicates(subset=['headline', 'url'], keep='first')
            removed = original - len(df)

            print("\n" + "=" * 70)
            print("Data Summary")
            print("=" * 70)
            print(f"Total: {len(df):,} articles (removed {removed:,} duplicates)")

            if 'data_source' in df.columns:
                print(f"\nData source distribution:")
                source_counts = df['data_source'].value_counts()
                for source, count in source_counts.items():
                    print(f"  {source:12s}: {count:,} articles")

            if 'symbol' in df.columns:
                stock_counts = df[df['symbol'].notna()]['symbol'].value_counts()
                if not stock_counts.empty:
                    print(f"\nStock distribution:")
                    for symbol, count in stock_counts.items():
                        print(f"  {symbol:6s}: {count:,} articles")

            print("=" * 70)
        else:
            print("\nNo data collected, please check the configuration")

        return df


    def save_data(self, df):
        """
        Save collected data

        Args:
            df: News DataFrame

        Returns:
            Path: Path to the saved JSON file
        """
        if df.empty:
            print("No data to save")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.teammate_id}_news_{timestamp}"

        # JSON format (primary file)
        json_path = self.output_dir / f"{filename}.json"
        df.to_json(json_path, orient='records', indent=2, force_ascii=False)

        # CSV format (for easy viewing)
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Statistics report
        stats_path = self.output_dir / f"{filename}_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"QianTick News Collection Statistics Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Collector: {self.config['name']} ({self.teammate_id})\n")
            f.write(f"Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if self.config['finnhub']:
                f.write(f"Finnhub Tasks:\n")
                f.write(f"  Stocks: {', '.join(self.config['stocks'])}\n")
                f.write(f"  Period: {DATA_START_DATE.date()} to {DATA_END_DATE.date()}\n\n")

            if self.config['gdelt']:
                f.write(f"GDELT Tasks:\n")
                f.write(f"  Period: {self.config['gdelt_start'].date()} to {self.config['gdelt_end'].date()}\n")
                f.write(f"  Number of themes: {len(GDELT_THEMES)}\n\n")

            f.write(f"{'='*60}\n\n")
            f.write(f"Collection Results:\n")
            f.write(f"  Total: {len(df):,} articles\n\n")

            if 'data_source' in df.columns:
                f.write(f"Data source distribution:\n")
                for source, count in df['data_source'].value_counts().items():
                    f.write(f"  {source}: {count:,} articles\n")
                f.write("\n")

            if 'symbol' in df.columns:
                stock_counts = df[df['symbol'].notna()]['symbol'].value_counts()
                if not stock_counts.empty:
                    f.write(f"Stock distribution:\n")
                    for symbol, count in stock_counts.items():
                        f.write(f"  {symbol}: {count:,} articles\n")
                    f.write("\n")

            if 'datetime' in df.columns:
                f.write(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}\n")

        # Print save info
        file_size_mb = json_path.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print("Data saved successfully")
        print("=" * 70)
        print(f"  JSON: {json_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  CSV:  {csv_path}")
        print(f"  Stats: {stats_path}")
        print("=" * 70)

        return json_path


# ============================================
# Main Function
# ============================================

def main():
    """Main function"""
    import sys

    # Get teammate_id from command line arguments
    if len(sys.argv) > 1:
        teammate_id = sys.argv[1]
    else:
        # Default to teammate1
        teammate_id = 'teammate1'
        print(f"No teammate ID specified, using default: {teammate_id}")
        print(f"Hint: python team_news_collector.py teammate2\n")

    try:
        # Create collector
        collector = TeamNewsCollector(teammate_id)

        # Collect data
        news_df = collector.collect_all()

        # Save data
        if not news_df.empty:
            saved_path = collector.save_data(news_df)

            print("\n" + "=" * 70)
            print("Task completed!")
            print("=" * 70)
            print(f"\nNext steps:")
            print(f"   1. Check the data file: {saved_path}")
            print(f"   2. Send the JSON file to Teammate 6 for merging")
            print(f"   3. Keep the CSV and stats files for your own reference\n")
        else:
            print("\nCollection failed, please check:")
            print("   1. Is FINNHUB_API_KEY correct in the .env file?")
            print("   2. Is the network connection working?")
            print("   3. Has the API quota been exhausted?\n")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
        print("Hint: You can re-run the script later to continue\n")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
