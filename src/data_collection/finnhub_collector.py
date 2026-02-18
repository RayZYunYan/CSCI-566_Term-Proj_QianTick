"""
QianTick Finnhub News Collector (Weekly Segmentation)
================================================
Data Source: Finnhub Company News API
Collection Method: Weekly segmentation to avoid data loss
Time Range: 2025-02-17 to 2026-02-17 (~12 months)
Stock Count: 15 Tech Stocks (3 per teammate)
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
from collections import Counter

load_dotenv()


# ============================================
# Configuration - Modify here to switch teammate
# ============================================

# Switch teammate by changing this line
CURRENT_TEAMMATE = 'teammate1'  # Change to 'teammate2', 'teammate3', etc.


# ============================================
# Team Configuration (Do not modify)
# ============================================

# Finnhub available date range (based on testing)
FINNHUB_START_DATE = datetime(2025, 2, 17)  # Earliest available
FINNHUB_END_DATE = datetime(2026, 2, 17)    # Today + buffer

# Team Task Assignment
TEAM_CONFIG = {
    'teammate1': {
        'name': 'Ray',
        'stocks': ['AAPL', 'MSFT', 'GOOGL'],
    },
    'teammate2': {
        'name': 'Yuxuan',
        'stocks': ['NVDA', 'QCOM', 'INTC'],
    },
    'teammate3': {
        'name': 'Yunfei Wang',
        'stocks': ['AMZN', 'META', 'AMD'],
    },
    'teammate4': {
        'name': 'Kaicheng Li',
        'stocks': ['TSLA', 'NFLX', 'CRM'],
    },
    'teammate5': {
        'name': 'Zeli Zhu',
        'stocks': ['ORCL', 'ADBE', 'SHOP'],
    },
    'teammate6': {
        'name': 'Hanzhen Qin',
        'stocks': [],  # No Finnhub task
    },
}


# ============================================
# Utility Functions
# ============================================

def generate_weekly_periods(start_date, end_date):
    """
    Generate weekly time periods
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        list: [(start, end), ...] list of weekly periods
    """
    periods = []
    current = start_date
    
    while current < end_date:
        week_end = current + timedelta(days=7)
        if week_end > end_date:
            week_end = end_date
        
        periods.append((current, week_end))
        current = week_end + timedelta(days=1)
    
    return periods


def format_time_remaining(seconds):
    """Format remaining time"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes} min {secs} sec"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hour {minutes} min"


# ============================================
# Finnhub Collector
# ============================================

class FinnhubCollector:
    """Finnhub News Collector with Weekly Segmentation"""
    
    def __init__(self, teammate_id):
        """Initialize collector"""
        if teammate_id not in TEAM_CONFIG:
            raise ValueError(f"Invalid teammate_id: {teammate_id}")
        
        self.teammate_id = teammate_id
        self.config = TEAM_CONFIG[teammate_id]
        
        # API Key
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        if self.config['stocks'] and not self.finnhub_key:
            raise ValueError("FINNHUB_API_KEY not found in .env file")
        
        # Output directory
        self.output_dir = Path(f"data/raw/news/{teammate_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate weekly periods
        self.weeks = generate_weekly_periods(FINNHUB_START_DATE, FINNHUB_END_DATE)
        
        self._print_header()
    
    
    def _print_header(self):
        """Print configuration info"""
        print("\n" + "=" * 70)
        print("🚀 Finnhub News Collector (Weekly Segmentation)")
        print("=" * 70)
        print(f"Teammate: {self.config['name']} ({self.teammate_id})")
        print(f"Assigned Stocks: {', '.join(self.config['stocks']) if self.config['stocks'] else 'None'}")
        print(f"Time Range: {FINNHUB_START_DATE.date()} to {FINNHUB_END_DATE.date()}")
        print(f"Total Weeks: {len(self.weeks)}")
        
        if self.config['stocks']:
            total_requests = len(self.config['stocks']) * len(self.weeks)
            estimated_time = total_requests * 1.2  # 1 sec per request + buffer
            print(f"Total Requests: {total_requests}")
            print(f"Estimated Time: {format_time_remaining(estimated_time)}")
        
        print("=" * 70 + "\n")
    
    
    def fetch_week_news(self, symbol, start_date, end_date):
        """
        Fetch news for one week
        
        Args:
            symbol: Stock ticker
            start_date: Week start date
            end_date: Week end date
            
        Returns:
            list: News articles
        """
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
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
            
        except Exception as e:
            print(f"\n      ✗ Request failed: {e}")
            return []
    
    
    def collect_stock_news(self, symbol):
        """
        Collect all news for one stock (all weeks)
        
        Args:
            symbol: Stock ticker
            
        Returns:
            list: All news for this stock
        """
        print(f"\n{'='*70}")
        print(f"Collecting {symbol}")
        print(f"{'='*70}")
        
        all_news = []
        start_time = time.time()
        
        for i, (week_start, week_end) in enumerate(self.weeks, 1):
            print(f"[{i}/{len(self.weeks)}] {week_start.date()} ~ {week_end.date()} ", end='', flush=True)
            
            news = self.fetch_week_news(symbol, week_start, week_end)
            all_news.extend(news)
            
            print(f"✓ {len(news)} articles", end='')
            
            # Show progress every 10 weeks
            if i % 10 == 0:
                elapsed = time.time() - start_time
                progress = (i / len(self.weeks)) * 100
                avg_time = elapsed / i
                remaining = avg_time * (len(self.weeks) - i)
                print(f" | Progress: {progress:.1f}% | Remaining: {format_time_remaining(remaining)}")
            else:
                print()
            
            # Wait 1 second (60 calls/min limit)
            if i < len(self.weeks):
                time.sleep(1)
        
        # Deduplicate
        unique_news = []
        seen_urls = set()
        for item in all_news:
            url = item.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_news.append(item)
        
        duplicates_removed = len(all_news) - len(unique_news)
        
        print(f"\n✅ {symbol} Complete:")
        print(f"   Total: {len(unique_news):,} articles (removed {duplicates_removed} duplicates)")
        
        # Show time distribution
        if unique_news:
            timestamps = [item['datetime'] for item in unique_news]
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            months = [d.strftime('%Y-%m') for d in dates]
            month_counts = Counter(months)
            
            print(f"   Time range: {min(dates).date()} to {max(dates).date()}")
            print(f"   Monthly distribution:")
            for month in sorted(month_counts.keys()):
                print(f"     {month}: {month_counts[month]:>4} articles")
        
        return unique_news
    
    
    def collect_all(self):
        """
        Collect news for all assigned stocks
        
        Returns:
            DataFrame: All collected news
        """
        if not self.config['stocks']:
            print("⚠️  This teammate has no Finnhub collection task")
            return pd.DataFrame()
        
        print("\n" + "🚀" * 35)
        print(f"Starting collection for {self.config['name']}")
        print("🚀" * 35)
        
        all_news = []
        overall_start = time.time()
        
        for idx, symbol in enumerate(self.config['stocks'], 1):
            print(f"\n[Stock {idx}/{len(self.config['stocks'])}]")
            
            stock_news = self.collect_stock_news(symbol)
            all_news.extend(stock_news)
            
            # Wait between stocks
            if idx < len(self.config['stocks']):
                print(f"\n⏸️  Waiting 3 seconds before next stock...\n")
                time.sleep(3)
        
        total_time = time.time() - overall_start
        
        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        
        if not df.empty:
            # Convert datetime
            df['datetime_unix'] = df['datetime']  # Keep original
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
            
            print("\n" + "=" * 70)
            print("📊 Collection Summary")
            print("=" * 70)
            print(f"Total Articles: {len(df):,}")
            print(f"Total Time: {format_time_remaining(total_time)}")
            
            # Stock distribution
            print(f"\nStock Distribution:")
            for symbol, count in df['symbol'].value_counts().items():
                print(f"  {symbol:6s}: {count:>5,} articles")
            
            # Time range
            print(f"\nTime Range:")
            print(f"  Earliest: {df['datetime'].min()}")
            print(f"  Latest: {df['datetime'].max()}")
            
            # Monthly distribution
            df['month'] = df['datetime'].dt.strftime('%Y-%m')
            print(f"\nMonthly Distribution:")
            for month, count in df['month'].value_counts().sort_index().items():
                print(f"  {month}: {count:>5,} articles")
            
            df = df.drop('month', axis=1)  # Remove temp column
            
            print("=" * 70)
        else:
            print("\n⚠️  No data collected")
        
        return df
    
    
    def save_data(self, df):
        """
        Save collected data
        
        Args:
            df: News DataFrame
            
        Returns:
            Path: Path to saved JSON file
        """
        if df.empty:
            print("❌ No data to save")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.teammate_id}_finnhub_{timestamp}"
        
        # Prepare data for JSON (convert datetime back to timestamp)
        df_export = df.copy()
        if 'datetime' in df_export.columns:
            df_export['datetime'] = df_export['datetime_unix']
            df_export = df_export.drop('datetime_unix', axis=1)
        
        # JSON format
        json_path = self.output_dir / f"{filename}.json"
        df_export.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        # CSV format
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Stats report
        stats_path = self.output_dir / f"{filename}_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"QianTick Finnhub Collection Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Collector: {self.config['name']} ({self.teammate_id})\n")
            f.write(f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Stocks: {', '.join(self.config['stocks'])}\n")
            f.write(f"  Time Range: {FINNHUB_START_DATE.date()} to {FINNHUB_END_DATE.date()}\n")
            f.write(f"  Weeks: {len(self.weeks)}\n\n")
            
            f.write(f"{'='*60}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Articles: {len(df):,}\n\n")
            
            f.write(f"Stock Distribution:\n")
            for symbol, count in df['symbol'].value_counts().items():
                f.write(f"  {symbol}: {count:,} articles\n")
            f.write("\n")
            
            f.write(f"Time Range:\n")
            f.write(f"  Earliest: {df['datetime'].min()}\n")
            f.write(f"  Latest: {df['datetime'].max()}\n\n")
            
            df_temp = df.copy()
            df_temp['month'] = df_temp['datetime'].dt.strftime('%Y-%m')
            f.write(f"Monthly Distribution:\n")
            for month, count in df_temp['month'].value_counts().sort_index().items():
                f.write(f"  {month}: {count:,} articles\n")
        
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        
        print("\n" + "=" * 70)
        print("💾 Data Saved")
        print("=" * 70)
        print(f"✓ JSON: {json_path.name} ({file_size_mb:.2f} MB)")
        print(f"✓ CSV:  {csv_path.name}")
        print(f"✓ Stats: {stats_path.name}")
        print("=" * 70)
        
        return json_path


# ============================================
# Main Function
# ============================================

def main():
    """Main function"""
    try:
        # Create collector
        collector = FinnhubCollector(CURRENT_TEAMMATE)
        
        # Collect data
        news_df = collector.collect_all()
        
        # Save data
        if not news_df.empty:
            saved_path = collector.save_data(news_df)
            
            print("\n" + "🎉" * 35)
            print("✅ Collection Complete!")
            print("🎉" * 35)
            print(f"\n📂 Data File: {saved_path}")
            print(f"📤 Next Steps:")
            print(f"   1. Check the data quality in the CSV file")
            print(f"   2. Share the JSON file with Teammate 6 for merging")
            print(f"   3. Keep stats file for your records\n")
        else:
            print("\n❌ Collection Failed")
            print("Please check:")
            print("   1. FINNHUB_API_KEY in .env file")
            print("   2. Network connection")
            print("   3. API quota\n")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("You can re-run the script to continue\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()