"""
Finnhub 按周分段测试脚本
测试范围：AAPL，2025年12月（4-5周）
"""

import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv('FINNHUB_API_KEY')

def generate_weekly_periods(start_date, end_date):
    """生成按周分段的时间列表"""
    periods = []
    current = start_date
    
    while current < end_date:
        week_end = current + timedelta(days=7)
        if week_end > end_date:
            week_end = end_date
        
        periods.append((current, week_end))
        current = week_end + timedelta(days=1)
    
    return periods


def fetch_finnhub_week(symbol, start_date, end_date):
    """获取单周数据"""
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': symbol,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': FINNHUB_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"      请求失败: {e}")
        return []


def main():
    """测试主函数"""
    print("=" * 70)
    print("Finnhub 按周分段测试")
    print("=" * 70)
    
    # 测试范围：2025年12月
    test_start = datetime(2025, 12, 1)
    test_end = datetime(2025, 12, 31)
    symbol = 'AAPL'
    
    print(f"\n测试股票: {symbol}")
    print(f"测试时间: {test_start.date()} 到 {test_end.date()}")
    
    # 生成周期
    weeks = generate_weekly_periods(test_start, test_end)
    print(f"分段数量: {len(weeks)} 周")
    print()
    
    all_news = []
    
    for i, (week_start, week_end) in enumerate(weeks, 1):
        print(f"[{i}/{len(weeks)}] {week_start.date()} ~ {week_end.date()} ", end='')
        
        news = fetch_finnhub_week(symbol, week_start, week_end)
        all_news.extend(news)
        
        print(f"✓ {len(news)} 条")
        
        # 显示这批数据的实际时间范围
        if news:
            timestamps = [item['datetime'] for item in news]
            earliest = datetime.fromtimestamp(min(timestamps))
            latest = datetime.fromtimestamp(max(timestamps))
            print(f"      实际范围: {earliest.date()} 到 {latest.date()}")
        
        # 等待1秒（60 calls/min = 每秒1次）
        if i < len(weeks):
            time.sleep(1)
    
    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"总计: {len(all_news)} 条新闻")
    
    if all_news:
        # 统计实际覆盖的日期
        timestamps = [item['datetime'] for item in all_news]
        earliest = datetime.fromtimestamp(min(timestamps))
        latest = datetime.fromtimestamp(max(timestamps))
        
        print(f"实际时间范围: {earliest.date()} 到 {latest.date()}")
        
        # 统计每天的新闻数
        from collections import Counter
        dates = [datetime.fromtimestamp(ts).date() for ts in timestamps]
        date_counts = Counter(dates)
        
        print(f"\n每日新闻数量:")
        for date in sorted(date_counts.keys()):
            print(f"  {date}: {date_counts[date]} 条")
    
    print()
    print("✅ 测试完成！")
    print("\n结论:")
    print("  - 如果覆盖了完整12月 → 按周分段有效")
    print("  - 如果只有部分日期 → 需要更密集分段（每3天或每天）")


if __name__ == "__main__":
    main()