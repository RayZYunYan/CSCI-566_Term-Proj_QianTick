"""
QianTick Team News Data Merge Script
================================================
Used to merge news data collected by all teammates
================================================
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json


def merge_all_team_news():
    """Merge all teammates' news data"""
    
    print("=" * 70)
    print("Starting team news data merge")
    print("=" * 70)
    
    base_dir = Path("data/raw/news")
    all_dfs = []
    
    # Iterate through all teammate directories
    teammate_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('teammate')])
    
    if not teammate_dirs:
        print("\nNo teammate data directories found")
        print("Hint: Data should be in data/raw/news/teammate1/, teammate2/, ...")
        return
    
    print(f"\nFound {len(teammate_dirs)} teammate directories\n")
    
    # Read each teammate's data
    for teammate_dir in teammate_dirs:
        teammate_id = teammate_dir.name
        
        # Find the latest JSON file
        json_files = list(teammate_dir.glob("*_news_*.json"))
        
        if not json_files:
            print(f"  {teammate_id}: No data files found, skipping")
            continue
        
        # Use the most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        file_size_mb = latest_file.stat().st_size / (1024 * 1024)
        
        # Read data
        try:
            df = pd.read_json(latest_file)
            all_dfs.append(df)
            
            print(f"  {teammate_id}:")
            print(f"    File: {latest_file.name}")
            print(f"    Size: {file_size_mb:.2f} MB")
            print(f"    Count: {len(df):,}")
            
        except Exception as e:
            print(f"  {teammate_id}: Read failed - {e}")
            continue
    
    if not all_dfs:
        print("\nFailed to read any data files")
        return
    
    # Merge all data
    print("\n" + "-" * 70)
    print("Merging data...")
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total before dedup: {len(merged_df):,}")
    
    # Deduplicate
    original = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['headline', 'url'], keep='first')
    removed = original - len(merged_df)
    
    print(f"   After dedup: {len(merged_df):,} (removed {removed:,} duplicates)")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Merged Data Statistics")
    print("=" * 70)
    
    print(f"\nTotal: {len(merged_df):,} articles")
    
    if 'data_source' in merged_df.columns:
        print(f"\nData source distribution:")
        for source, count in merged_df['data_source'].value_counts().items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {source:12s}: {count:>8,} articles ({percentage:>5.1f}%)")
    
    if 'collected_by' in merged_df.columns:
        print(f"\nCollector distribution:")
        for collector, count in merged_df['collected_by'].value_counts().items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {collector:12s}: {count:>8,} articles ({percentage:>5.1f}%)")
    
    if 'symbol' in merged_df.columns:
        stock_counts = merged_df[merged_df['symbol'].notna()]['symbol'].value_counts()
        if not stock_counts.empty:
            print(f"\nStock distribution (Top 10):")
            for symbol, count in stock_counts.head(10).items():
                print(f"  {symbol:6s}: {count:>6,} articles")
    
    if 'theme' in merged_df.columns:
        theme_counts = merged_df[merged_df['theme'].notna()]['theme'].value_counts()
        if not theme_counts.empty:
            print(f"\nGDELT theme distribution (Top 5):")
            for theme, count in theme_counts.head(5).items():
                theme_short = theme[:50] + "..." if len(theme) > 50 else theme
                print(f"  {theme_short:53s}: {count:>6,} articles")
    
    if 'datetime' in merged_df.columns:
        print(f"\nTime range:")
        print(f"  Earliest: {merged_df['datetime'].min()}")
        print(f"  Latest:   {merged_df['datetime'].max()}")
    
    # Save merged data
    print("\n" + "=" * 70)
    print("Saving merged data")
    print("=" * 70)
    
    output_dir = base_dir / "merged"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON format
    json_path = output_dir / f"qiantick_all_news_{timestamp}.json"
    merged_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    json_size_mb = json_path.stat().st_size / (1024 * 1024)
    print(f"  JSON: {json_path}")
    print(f"  Size: {json_size_mb:.2f} MB")
    
    # CSV format
    csv_path = output_dir / f"qiantick_all_news_{timestamp}.csv"
    merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  CSV:  {csv_path}")
    print(f"  Size: {csv_size_mb:.2f} MB")
    
    # Save detailed statistics report
    stats_path = output_dir / f"qiantick_all_news_{timestamp}_report.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("QianTick Project - News Data Merge Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Merge time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Teammates involved: {len(all_dfs)}\n")
        f.write(f"Total articles: {len(merged_df):,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Data Source Distribution\n")
        f.write("=" * 70 + "\n")
        if 'data_source' in merged_df.columns:
            for source, count in merged_df['data_source'].value_counts().items():
                percentage = (count / len(merged_df)) * 100
                f.write(f"{source:15s}: {count:>8,} articles ({percentage:>5.1f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Collector Distribution\n")
        f.write("=" * 70 + "\n")
        if 'collected_by' in merged_df.columns:
            for collector, count in merged_df['collected_by'].value_counts().items():
                percentage = (count / len(merged_df)) * 100
                f.write(f"{collector:15s}: {count:>8,} articles ({percentage:>5.1f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Stock Distribution (Finnhub Data)\n")
        f.write("=" * 70 + "\n")
        if 'symbol' in merged_df.columns:
            stock_counts = merged_df[merged_df['symbol'].notna()]['symbol'].value_counts()
            for symbol, count in stock_counts.items():
                f.write(f"{symbol:8s}: {count:>6,} articles\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("GDELT Theme Distribution\n")
        f.write("=" * 70 + "\n")
        if 'theme' in merged_df.columns:
            theme_counts = merged_df[merged_df['theme'].notna()]['theme'].value_counts()
            for theme, count in theme_counts.items():
                f.write(f"{theme:60s}: {count:>6,} articles\n")
        
        if 'datetime' in merged_df.columns:
            f.write("\n" + "=" * 70 + "\n")
            f.write("Time Range\n")
            f.write("=" * 70 + "\n")
            f.write(f"Earliest: {merged_df['datetime'].min()}\n")
            f.write(f"Latest:   {merged_df['datetime'].max()}\n")
    
    print(f"  Report: {stats_path}")
    
    # Data quality check
    print("\n" + "=" * 70)
    print("Data Quality Check")
    print("=" * 70)
    
    # Check for missing values
    missing = merged_df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing value statistics:")
        for col, count in missing[missing > 0].items():
            percentage = (count / len(merged_df)) * 100
            print(f"  {col:15s}: {count:>6,} ({percentage:>5.1f}%)")
    else:
        print("\n  No missing values")
    
    # Check for duplicate URLs
    duplicate_urls = merged_df[merged_df.duplicated(subset=['url'], keep=False)]
    if len(duplicate_urls) > 0:
        print(f"\n  Found {len(duplicate_urls)} duplicate URLs (handled during deduplication)")
    
    print("\n" + "🎉" * 35)
    print("Data merge completed!")
    print("🎉" * 35)
    print(f"\nMerged files:")
    print(f"   JSON:   {json_path}")
    print(f"   CSV:    {csv_path}")
    print(f"   Report: {stats_path}")
    print(f"\nData scale:")
    print(f"   Article count: {len(merged_df):,}")
    print(f"   File size: {json_size_mb:.2f} MB (JSON) + {csv_size_mb:.2f} MB (CSV)")
    print(f"\nNext steps:")
    print(f"   1. Review the data quality report")
    print(f"   2. Start data preprocessing (src/preprocessing/)")
    print(f"   3. Run exploratory data analysis (notebooks/01_EDA.ipynb)\n")
    
    return merged_df


if __name__ == "__main__":
    merge_all_team_news()