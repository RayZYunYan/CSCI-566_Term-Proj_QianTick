"""
QianTick GDELT News Collector - Final Version
================================================
Features:
- Monthly segmentation (avoids data concentration)
- English-only articles (sourcelang:eng)
- Improved deduplication (by headline only)
- 18 themes covering macro finance and tech
- Automatic retry mechanism
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
# Configuration - Modify here to switch teammate
# ============================================

# Switch teammate by changing this line
CURRENT_TEAMMATE = 'teammate_'  # Available teammates: teammate1, teammate2, teammate3, teammate4, teammate5, teammate6


# ============================================
# Team Configuration (Do not modify)
# ============================================

# Team Task Assignment (GDELT time periods)
TEAM_CONFIG = {
    'teammate1': {
        'name': 'Ray',
        'gdelt_start': datetime(2022, 6, 1),
        'gdelt_end': datetime(2023, 1, 31),
    },
    'teammate2': {
        'name': 'Teammate 2',
        'gdelt_start': datetime(2023, 2, 1),
        'gdelt_end': datetime(2023, 8, 31),
    },
    'teammate3': {
        'name': 'Teammate 3',
        'gdelt_start': datetime(2023, 9, 1),
        'gdelt_end': datetime(2024, 3, 31),
    },
    'teammate4': {
        'name': 'Teammate 4',
        'gdelt_start': datetime(2024, 4, 1),
        'gdelt_end': datetime(2024, 10, 31),
    },
    'teammate5': {
        'name': 'Teammate 5',
        'gdelt_start': datetime(2024, 11, 1),
        'gdelt_end': datetime(2025, 5, 31),
    },
    'teammate6': {
        'name': 'Teammate 6',
        'gdelt_start': datetime(2025, 6, 1),
        'gdelt_end': datetime(2025, 12, 31),
    },
}

# GDELT Macro News Themes (18 themes - English only)
GDELT_THEMES = [
    # Original successful themes (10)
    "Federal Reserve interest rate technology",
    "Fed policy tech stocks",
    "technology regulation policy",
    "antitrust tech companies",
    "CHIPS act semiconductor",
    "chip shortage supply chain",
    "inflation technology sector",
    "recession tech stocks",
    "semiconductor geopolitics",
    "Silicon Valley",
    
    # New financial/Wall Street themes (8)
    "tech stocks Wall Street",
    "tech sector earnings",
    "technology stocks analyst",
    "venture capital technology",
    "tech IPO nasdaq",
    "technology investment market",
    "tech stocks bull market",
    "technology sector outlook",
]


# ============================================
# Utility Functions
# ============================================

def generate_monthly_periods(start_date, end_date):
    """
    Generate monthly time periods
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        list: [(month_start, month_end, 'YYYY-MM'), ...]
    """
    periods = []
    current = start_date.replace(day=1)  # Start of month
    
    while current <= end_date:
        # Calculate month end
        if current.month == 12:
            month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
        
        # Don't exceed end_date
        if month_end > end_date:
            month_end = end_date
        
        month_label = current.strftime('%Y-%m')
        periods.append((current, month_end, month_label))
        
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return periods


def format_time_remaining(seconds):
    """Format remaining time"""
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes} min {secs} sec"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hour {minutes} min"


# ============================================
# GDELT Collector
# ============================================

class GDELTCollector:
    """GDELT News Collector - Final Version"""
    
    def __init__(self, teammate_id):
        """Initialize collector"""
        if teammate_id not in TEAM_CONFIG:
            raise ValueError(f"Invalid teammate_id: {teammate_id}")
        
        self.teammate_id = teammate_id
        self.config = TEAM_CONFIG[teammate_id]
        
        # Output directory (always relative to project root, not cwd)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = project_root / "data" / "raw" / "news" / teammate_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate monthly periods
        self.months = generate_monthly_periods(
            self.config['gdelt_start'],
            self.config['gdelt_end']
        )
        
        # Track failed requests for retry
        self.failed_requests = []
        
        self._print_header()
    
    
    def _print_header(self):
        """Print configuration info"""
        gdelt_start = self.config['gdelt_start']
        gdelt_end = self.config['gdelt_end']
        
        print("\n" + "=" * 70)
        print("🌐 GDELT Macro News Collector - FINAL VERSION")
        print("=" * 70)
        print(f"Teammate: {self.config['name']} ({self.teammate_id})")
        print(f"Time Period: {gdelt_start.date()} to {gdelt_end.date()}")
        print(f"Duration: {len(self.months)} months")
        print(f"Themes: {len(GDELT_THEMES)}")
        print(f"Strategy: Monthly segmentation + headline deduplication")
        print(f"Language: English only")
        
        total_requests = len(GDELT_THEMES) * len(self.months)
        estimated_time = total_requests * 4
        
        print(f"\nTotal Requests: {total_requests}")
        print(f"Estimated Time: {format_time_remaining(estimated_time)}")
        print(f"Estimated Articles: ~{len(self.months) * len(GDELT_THEMES) * 100:,} unique")
        print("=" * 70 + "\n")
    
    
    def fetch_gdelt_month(self, theme, month_start, month_end, month_label, max_retries=3):
        """
        Fetch news for one theme in one month
        
        Args:
            theme: Theme keywords
            month_start: Month start date
            month_end: Month end date
            month_label: Month label (YYYY-MM)
            max_retries: Max retry attempts
            
        Returns:
            list: News articles
        """
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        params = {
            'query': f'{theme} sourcelang:eng',  # English only
            'mode': 'artlist',
            'maxrecords': 250,
            'startdatetime': month_start.strftime('%Y%m%d000000'),
            'enddatetime': month_end.strftime('%Y%m%d235959'),
            'format': 'json',
            'sort': 'datedesc'
        }
        
        for attempt in range(max_retries):
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
                            'theme': theme,
                            'month_collected': month_label,
                            'data_source': 'gdelt',
                            'collected_by': self.teammate_id
                        })
                
                return articles
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\n      ⚠️  Timeout, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...", end='')
                    time.sleep(wait_time)
                else:
                    print(f"\n      ✗ Failed after {max_retries} attempts (Timeout)")
                    return []
            
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\n      ⚠️  Request error, retrying in {wait_time}s...", end='')
                    time.sleep(wait_time)
                else:
                    print(f"\n      ✗ Failed: {str(e)[:50]}")
                    return []
            
            except (json.JSONDecodeError, TypeError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\n      ⚠️  Parse error, retrying in {wait_time}s...", end='')
                    time.sleep(wait_time)
                else:
                    print(f"\n      ✗ Parse error: {str(e)[:50]}")
                    return []
        
        return []
    
    
    def collect_all(self):
        """
        Collect news for all themes (monthly segmentation)
        
        Returns:
            DataFrame: All collected news
        """
        print("\n" + "🚀" * 35)
        print(f"Starting GDELT collection for {self.config['name']}")
        print("🚀" * 35 + "\n")
        
        all_news = []
        start_time = time.time()
        
        total_requests = len(GDELT_THEMES) * len(self.months)
        completed = 0
        
        # Collect by theme, then by month
        for theme_idx, theme in enumerate(GDELT_THEMES, 1):
            theme_short = theme[:45] + "..." if len(theme) > 45 else theme
            
            print(f"\n{'='*70}")
            print(f"[Theme {theme_idx}/{len(GDELT_THEMES)}] {theme_short}")
            print(f"{'='*70}")
            
            theme_news = []
            
            for month_idx, (month_start, month_end, month_label) in enumerate(self.months, 1):
                print(f"  [{month_idx}/{len(self.months)}] {month_label} ", end='', flush=True)
                
                news = self.fetch_gdelt_month(theme, month_start, month_end, month_label)
                
                if news:
                    theme_news.extend(news)
                    print(f"✓ {len(news)} articles")
                else:
                    print(f"✗ 0 articles")
                    self.failed_requests.append((theme, month_start, month_end, month_label))
                
                completed += 1
                
                # Show progress every 6 requests
                if completed % 6 == 0:
                    elapsed = time.time() - start_time
                    progress = (completed / total_requests) * 100
                    avg_time = elapsed / completed
                    remaining = avg_time * (total_requests - completed)
                    print(f"      Progress: {progress:.1f}% | Remaining: {format_time_remaining(remaining)}")
                
                # Wait between requests
                if completed < total_requests:
                    time.sleep(4)
            
            all_news.extend(theme_news)
            print(f"  ✓ {theme_short}: {len(theme_news)} articles")
        
        # Retry failed requests
        if self.failed_requests:
            print(f"\n{'='*70}")
            print(f"⚠️  Retrying {len(self.failed_requests)} failed requests")
            print(f"{'='*70}\n")
            
            for i, (theme, month_start, month_end, month_label) in enumerate(self.failed_requests, 1):
                theme_short = theme[:45] + "..." if len(theme) > 45 else theme
                print(f"[Retry {i}/{len(self.failed_requests)}] {theme_short} {month_label} ", end='', flush=True)
                
                time.sleep(10)
                
                news = self.fetch_gdelt_month(theme, month_start, month_end, month_label, max_retries=5)
                
                if news:
                    all_news.extend(news)
                    print(f"✓ {len(news)} articles (Recovered)")
                else:
                    print(f"✗ Still failed")
                
                if i < len(self.failed_requests):
                    time.sleep(6)
        
        total_time = time.time() - start_time
        
        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        
        if not df.empty:
            # Deduplicate by headline only (removes syndicated duplicates)
            original = len(df)
            
            print(f"\n{'='*70}")
            print("🔄 Deduplicating (by headline only)...")
            print(f"{'='*70}")
            print(f"Before dedup: {original:,} articles")
            
            df = df.drop_duplicates(subset=['headline'], keep='first')
            removed = original - len(df)
            
            print(f"After dedup:  {len(df):,} articles")
            print(f"Removed:      {removed:,} duplicates ({(removed/original)*100:.1f}%)")
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            print(f"\n{'='*70}")
            print("📊 Collection Summary")
            print(f"{'='*70}")
            print(f"Final Count: {len(df):,} unique articles")
            print(f"Total Time: {format_time_remaining(total_time)}")
            
            # Theme distribution
            print(f"\nTheme Distribution:")
            for theme, count in df['theme'].value_counts().items():
                theme_clean = theme.replace(' sourcelang:eng', '')
                theme_short = theme_clean[:45] + "..." if len(theme_clean) > 45 else theme_clean
                print(f"  {theme_short:48s}: {count:>5,} articles")
            
            # Monthly distribution
            print(f"\nMonthly Distribution:")
            for month, count in df['month_collected'].value_counts().sort_index().items():
                print(f"  {month}: {count:>5,} articles")
            
            # Time range
            print(f"\nActual Time Range:")
            print(f"  Earliest: {df['datetime'].min()}")
            print(f"  Latest: {df['datetime'].max()}")
            
            # Language distribution
            if 'language' in df.columns:
                print(f"\nLanguage Distribution:")
                for lang, count in df['language'].value_counts().head(5).items():
                    percentage = (count / len(df)) * 100
                    print(f"  {lang}: {count:,} articles ({percentage:.1f}%)")
            
            print(f"{'='*70}")
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
        filename = f"{self.teammate_id}_gdelt_{timestamp}"
        
        # Prepare data for JSON
        df_export = df.copy()
        
        # Clean theme names
        if 'theme' in df_export.columns:
            df_export['theme'] = df_export['theme'].str.replace(' sourcelang:eng', '', regex=False)
        
        # Convert datetime to string
        if 'datetime' in df_export.columns:
            df_export['datetime'] = df_export['datetime'].dt.strftime('%Y%m%d%H%M%S')
        
        # Remove temporary column
        if 'month_collected' in df_export.columns:
            df_export = df_export.drop('month_collected', axis=1)
        
        # JSON format
        json_path = self.output_dir / f"{filename}.json"
        df_export.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        # CSV format
        csv_path = self.output_dir / f"{filename}.csv"
        df_csv = df.copy()
        if 'theme' in df_csv.columns:
            df_csv['theme'] = df_csv['theme'].str.replace(' sourcelang:eng', '', regex=False)
        df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Stats report
        stats_path = self.output_dir / f"{filename}_stats.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"QianTick GDELT Collection Report - Final Version\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Collector: {self.config['name']} ({self.teammate_id})\n")
            f.write(f"Collection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Time Period: {self.config['gdelt_start'].date()} to {self.config['gdelt_end'].date()}\n")
            f.write(f"  Months: {len(self.months)}\n")
            f.write(f"  Themes: {len(GDELT_THEMES)}\n")
            f.write(f"  Strategy: Monthly segmentation + headline deduplication\n")
            f.write(f"  Language: English only\n\n")
            
            f.write(f"{'='*60}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total Articles: {len(df):,}\n\n")
            
            f.write(f"Theme Distribution:\n")
            df_temp = df.copy()
            df_temp['theme'] = df_temp['theme'].str.replace(' sourcelang:eng', '', regex=False)
            for theme, count in df_temp['theme'].value_counts().items():
                f.write(f"  {theme}: {count:,} articles\n")
            f.write("\n")
            
            f.write(f"Time Range:\n")
            f.write(f"  Earliest: {df['datetime'].min()}\n")
            f.write(f"  Latest: {df['datetime'].max()}\n\n")
            
            if 'language' in df.columns:
                f.write(f"Language Distribution:\n")
                for lang, count in df['language'].value_counts().items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  {lang}: {count:,} articles ({percentage:.1f}%)\n")
                f.write("\n")
            
            if 'month_collected' in df.columns:
                f.write(f"Monthly Distribution:\n")
                for month, count in df['month_collected'].value_counts().sort_index().items():
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
        collector = GDELTCollector(CURRENT_TEAMMATE)
        
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
            print(f"   1. Verify data quality in CSV file")
            print(f"   2. Share JSON file with Teammate 6 for merging")
            print(f"   3. Other teammates can change CURRENT_TEAMMATE and run\n")
        else:
            print("\n❌ Collection Failed")
            print("All requests failed to collect data\n")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("You can re-run the script to continue\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()