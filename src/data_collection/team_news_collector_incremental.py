"""
QianTick 增量新闻采集器（支持分段采集和断点续传）
================================================
改进功能：
1. Finnhub按季度分段采集（解决单次请求限制）
2. 自动检测已有数据，只采集缺失部分
3. 实时保存，防止中断丢失数据
4. 显示详细进度和预计剩余时间
5. 支持GDELT重新采集并去重
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
from collections import defaultdict

load_dotenv()


# ============================================
# 配置区域 - 只需修改这里
# ============================================

# 切换teammate只需改这一行
CURRENT_TEAMMATE = 'teammate1'  # 改成 'teammate2', 'teammate3' 等

# ============================================
# 团队配置（不需要改）
# ============================================

ALL_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL',
    'NVDA', 'QCOM', 'INTC',
    'AMZN', 'META', 'AMD',
    'TSLA', 'NFLX', 'CRM',
    'ORCL', 'ADBE', 'SHOP',
]

DATA_START_DATE = datetime(2025, 2, 15)
DATA_END_DATE = datetime(2025, 2, 15)

TEAM_CONFIG = {
    'teammate1': {
        'name': 'Ray',
        'stocks': ['AAPL', 'MSFT', 'GOOGL'],
        'gdelt_start': datetime(2022, 6, 1),
        'gdelt_end': datetime(2023, 1, 31),
    },
    'teammate2': {
        'name': 'Teammate 2',
        'stocks': ['NVDA', 'QCOM', 'INTC'],
        'gdelt_start': datetime(2023, 2, 1),
        'gdelt_end': datetime(2023, 8, 31),
    },
    'teammate3': {
        'name': 'Teammate 3',
        'stocks': ['AMZN', 'META', 'AMD'],
        'gdelt_start': datetime(2023, 9, 1),
        'gdelt_end': datetime(2024, 3, 31),
    },
    'teammate4': {
        'name': 'Teammate 4',
        'stocks': ['TSLA', 'NFLX', 'CRM'],
        'gdelt_start': datetime(2024, 4, 1),
        'gdelt_end': datetime(2024, 10, 31),
    },
    'teammate5': {
        'name': 'Teammate 5',
        'stocks': ['ORCL', 'ADBE', 'SHOP'],
        'gdelt_start': datetime(2024, 11, 1),
        'gdelt_end': datetime(2025, 5, 31),
    },
    'teammate6': {
        'name': 'Teammate 6',
        'stocks': [],
        'gdelt_start': datetime(2025, 6, 1),
        'gdelt_end': datetime(2025, 12, 31),
    },
}

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
# 工具函数
# ============================================

def generate_quarters(start_date, end_date):
    """
    生成季度列表
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        list: [(start, end), ...] 季度时间段列表
    """
    quarters = []
    current = start_date
    
    while current < end_date:
        # 计算季度结束日期（3个月后）
        quarter_end = current + timedelta(days=90)
        if quarter_end > end_date:
            quarter_end = end_date
        
        quarters.append((current, quarter_end))
        current = quarter_end + timedelta(days=1)
    
    return quarters


def format_time_remaining(seconds):
    """格式化剩余时间"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds/60)}分钟"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}小时{minutes}分钟"


# ============================================
# 增量采集器
# ============================================

class IncrementalNewsCollector:
    """支持增量采集和断点续传的新闻采集器"""
    
    def __init__(self, teammate_id):
        """初始化"""
        if teammate_id not in TEAM_CONFIG:
            raise ValueError(f"无效的teammate_id: {teammate_id}")
        
        self.teammate_id = teammate_id
        self.config = TEAM_CONFIG[teammate_id]
        
        # API Key
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        if self.config['stocks'] and not self.finnhub_key:
            raise ValueError("未找到FINNHUB_API_KEY")
        
        # 目录
        self.output_dir = Path(f"data/raw/news/{teammate_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self.existing_data = self._load_existing_data()
        
        self._print_header()
    
    
    def _load_existing_data(self):
        """加载已有的数据文件"""
        json_files = list(self.output_dir.glob("*_news_*.json"))
        
        if not json_files:
            print(f"📂 未找到已有数据文件")
            return pd.DataFrame()
        
        # 读取最新的文件
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        try:
            df = pd.read_json(latest_file)
            print(f"📂 加载已有数据: {latest_file.name}")
            print(f"   已有新闻: {len(df):,} 条")
            
            if 'data_source' in df.columns:
                print(f"   数据源分布:")
                for source, count in df['data_source'].value_counts().items():
                    print(f"     {source}: {count:,} 条")
            
            return df
            
        except Exception as e:
            print(f"⚠️  加载数据失败: {e}")
            return pd.DataFrame()
    
    
    def _print_header(self):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print(f"🚀 增量新闻采集器")
        print("=" * 70)
        print(f"队友: {self.config['name']} ({self.teammate_id})")
        print(f"负责股票: {', '.join(self.config['stocks']) if self.config['stocks'] else '无'}")
        print(f"GDELT时段: {self.config['gdelt_start'].date()} 到 {self.config['gdelt_end'].date()}")
        print("=" * 70 + "\n")
    
    
    # ========================================
    # Finnhub 分段采集
    # ========================================
    
    def _detect_missing_quarters(self, symbol, quarters):
        """
        检测缺失的季度数据
        
        Args:
            symbol: 股票代码
            quarters: 所有季度列表
            
        Returns:
            list: 需要采集的季度列表
        """
        if self.existing_data.empty:
            return quarters  # 没有旧数据，全部采集
        
        # 获取该股票的已有数据
        symbol_data = self.existing_data[
            (self.existing_data.get('symbol') == symbol) &
            (self.existing_data.get('data_source') == 'finnhub')
        ]
        
        if symbol_data.empty:
            return quarters  # 该股票没有数据，全部采集
        
        # 转换datetime为日期
        symbol_data['date'] = pd.to_datetime(symbol_data['datetime'], unit='s', errors='coerce')
        existing_dates = set(symbol_data['date'].dt.date)
        
        # 检查每个季度是否有数据
        missing_quarters = []
        for q_start, q_end in quarters:
            # 检查这个季度的日期范围内是否有数据
            quarter_dates = pd.date_range(q_start, q_end, freq='D')
            has_data = any(d.date() in existing_dates for d in quarter_dates[:30])  # 检查前30天
            
            if not has_data:
                missing_quarters.append((q_start, q_end))
        
        return missing_quarters
    
    
    def fetch_finnhub_quarter(self, symbol, start_date, end_date):
        """获取单个季度的Finnhub数据"""
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
            print(f"      ✗ 请求失败: {e}")
            return []
    
    
    def fetch_all_finnhub_incremental(self):
        """增量采集所有Finnhub数据"""
        if not self.config['stocks']:
            print("⚠️  此队友不负责Finnhub采集")
            return []
        
        print("\n" + "=" * 70)
        print("📊 开始增量采集 Finnhub 数据")
        print("=" * 70)
        
        # 生成所有季度
        all_quarters = generate_quarters(DATA_START_DATE, DATA_END_DATE)
        print(f"时间范围: {DATA_START_DATE.date()} 到 {DATA_END_DATE.date()}")
        print(f"总季度数: {len(all_quarters)}")
        
        all_new_news = []
        stocks = self.config['stocks']
        
        total_requests = 0
        for symbol in stocks:
            # 检测缺失的季度
            missing_quarters = self._detect_missing_quarters(symbol, all_quarters)
            total_requests += len(missing_quarters)
        
        print(f"预计请求次数: {total_requests}")
        print(f"预计耗时: {format_time_remaining(total_requests * 4)}")
        print()
        
        start_time = time.time()
        completed_requests = 0
        
        for stock_idx, symbol in enumerate(stocks, 1):
            print(f"\n{'='*70}")
            print(f"[{stock_idx}/{len(stocks)}] 正在采集 {symbol}")
            print(f"{'='*70}")
            
            # 检测缺失季度
            missing_quarters = self._detect_missing_quarters(symbol, all_quarters)
            
            if not missing_quarters:
                print(f"✓ {symbol} 数据已完整，跳过")
                continue
            
            print(f"需要采集: {len(missing_quarters)} 个季度（共{len(all_quarters)}个）")
            
            stock_news = []
            
            for q_idx, (q_start, q_end) in enumerate(missing_quarters, 1):
                print(f"  [{q_idx}/{len(missing_quarters)}] {q_start.date()} ~ {q_end.date()} ", end='')
                
                news = self.fetch_finnhub_quarter(symbol, q_start, q_end)
                stock_news.extend(news)
                
                print(f"✓ {len(news)} 条")
                
                completed_requests += 1
                
                # 计算剩余时间
                elapsed = time.time() - start_time
                avg_time_per_request = elapsed / completed_requests
                remaining_requests = total_requests - completed_requests
                eta = avg_time_per_request * remaining_requests
                
                if completed_requests % 5 == 0:  # 每5次显示一次进度
                    progress = (completed_requests / total_requests) * 100
                    print(f"      进度: {progress:.1f}% | 剩余时间: {format_time_remaining(eta)}")
                
                # 等待4秒
                if q_idx < len(missing_quarters):
                    time.sleep(4)
            
            all_new_news.extend(stock_news)
            print(f"\n  ✓ {symbol} 完成: 新增 {len(stock_news):,} 条新闻")
        
        print(f"\n{'='*70}")
        print(f"✅ Finnhub采集完成")
        print(f"   新增: {len(all_new_news):,} 条")
        print(f"   总耗时: {format_time_remaining(time.time() - start_time)}")
        print(f"{'='*70}")
        
        return all_new_news
    
    
    # ========================================
    # GDELT 采集
    # ========================================
    
    def fetch_gdelt_news(self, keywords, start_date, end_date, max_records=250):
        """获取GDELT新闻"""
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
            
        except Exception as e:
            print(f"      ✗ 请求失败: {e}")
            return []
    
    
    def fetch_all_gdelt(self):
        """采集所有GDELT数据"""
        print("\n" + "=" * 70)
        print("🌐 开始采集 GDELT 宏观新闻")
        print("=" * 70)
        
        gdelt_start = self.config['gdelt_start']
        gdelt_end = self.config['gdelt_end']
        
        print(f"时间范围: {gdelt_start.date()} 到 {gdelt_end.date()}")
        print(f"主题数量: {len(GDELT_THEMES)}")
        print(f"预计耗时: {format_time_remaining(len(GDELT_THEMES) * 2)}")
        print()
        
        all_news = []
        start_time = time.time()
        
        for i, theme in enumerate(GDELT_THEMES, 1):
            theme_short = theme[:50] + "..." if len(theme) > 50 else theme
            print(f"[{i}/{len(GDELT_THEMES)}] {theme_short} ", end='')
            
            news = self.fetch_gdelt_news(theme, gdelt_start, gdelt_end)
            all_news.extend(news)
            
            print(f"✓ {len(news):,} 条")
            
            if i < len(GDELT_THEMES):
                time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"✅ GDELT采集完成")
        print(f"   新增: {len(all_news):,} 条")
        print(f"   总耗时: {format_time_remaining(time.time() - start_time)}")
        print(f"{'='*70}")
        
        return all_news
    
    
    # ========================================
    # 数据合并和保存
    # ========================================
    
    def merge_and_save(self, new_finnhub_data, new_gdelt_data):
        """合并新旧数据并保存"""
        print("\n" + "=" * 70)
        print("🔄 合并数据")
        print("=" * 70)
        
        # 转换为DataFrame
        new_finnhub_df = pd.DataFrame(new_finnhub_data) if new_finnhub_data else pd.DataFrame()
        new_gdelt_df = pd.DataFrame(new_gdelt_data) if new_gdelt_data else pd.DataFrame()
        
        # 合并新数据
        new_data = pd.concat([new_finnhub_df, new_gdelt_df], ignore_index=True)
        
        print(f"新采集数据: {len(new_data):,} 条")
        
        if self.existing_data.empty:
            merged_df = new_data
            print(f"无旧数据，直接使用新数据")
        else:
            # 合并新旧数据
            merged_df = pd.concat([self.existing_data, new_data], ignore_index=True)
            print(f"旧数据: {len(self.existing_data):,} 条")
            print(f"合并前: {len(merged_df):,} 条")
            
            # 去重（按headline + url）
            original = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=['headline', 'url'], keep='first')
            removed = original - len(merged_df)
            
            print(f"去重后: {len(merged_df):,} 条（删除 {removed:,} 条重复）")
        
        # 统计
        print(f"\n最终数据统计:")
        print(f"  总计: {len(merged_df):,} 条")
        
        if 'data_source' in merged_df.columns:
            print(f"\n  数据源分布:")
            for source, count in merged_df['data_source'].value_counts().items():
                percentage = (count / len(merged_df)) * 100
                print(f"    {source:12s}: {count:>6,} 条 ({percentage:>5.1f}%)")
        
        if 'symbol' in merged_df.columns:
            stock_counts = merged_df[merged_df['symbol'].notna()]['symbol'].value_counts()
            if not stock_counts.empty:
                print(f"\n  股票分布:")
                for symbol, count in stock_counts.items():
                    print(f"    {symbol:6s}: {count:>5,} 条")
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.teammate_id}_news_{timestamp}"
        
        json_path = self.output_dir / f"{filename}.json"
        csv_path = self.output_dir / f"{filename}.csv"
        stats_path = self.output_dir / f"{filename}_stats.txt"
        
        merged_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 统计文件
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"QianTick News Collection Statistics Report\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Collector: {self.config['name']} ({self.teammate_id})\n")
            f.write(f"Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.config['stocks']:
                f.write(f"Finnhub Tasks:\n")
                f.write(f"  Stocks: {', '.join(self.config['stocks'])}\n")
                f.write(f"  Period: {DATA_START_DATE.date()} to {DATA_END_DATE.date()}\n\n")
            
            f.write(f"GDELT Tasks:\n")
            f.write(f"  Period: {self.config['gdelt_start'].date()} to {self.config['gdelt_end'].date()}\n")
            f.write(f"  Number of themes: {len(GDELT_THEMES)}\n\n")
            
            f.write(f"{'='*60}\n\n")
            f.write(f"Collection Results:\n")
            f.write(f"  Total: {len(merged_df):,} articles\n\n")
            
            if 'data_source' in merged_df.columns:
                f.write(f"Data source distribution:\n")
                for source, count in merged_df['data_source'].value_counts().items():
                    f.write(f"  {source}: {count:,} articles\n")
                f.write("\n")
            
            if 'symbol' in merged_df.columns:
                stock_counts = merged_df[merged_df['symbol'].notna()]['symbol'].value_counts()
                if not stock_counts.empty:
                    f.write(f"Stock distribution:\n")
                    for symbol, count in stock_counts.items():
                        f.write(f"  {symbol}: {count:,} articles\n")
        
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"💾 数据已保存")
        print(f"{'='*70}")
        print(f"✓ JSON: {json_path.name} ({file_size_mb:.2f} MB)")
        print(f"✓ CSV:  {csv_path.name}")
        print(f"✓ 统计: {stats_path.name}")
        print(f"{'='*70}")
        
        return json_path
    
    
    # ========================================
    # 主流程
    # ========================================
    
    def run(self):
        """执行采集任务"""
        print("\n" + "🚀" * 35)
        print(f"开始执行 {self.config['name']} 的增量采集任务")
        print("🚀" * 35)
        
        try:
            # 1. Finnhub增量采集
            new_finnhub = []
            if self.config['stocks']:
                new_finnhub = self.fetch_all_finnhub_incremental()
            
            # 2. GDELT采集
            new_gdelt = self.fetch_all_gdelt()
            
            # 3. 合并保存
            if new_finnhub or new_gdelt:
                saved_path = self.merge_and_save(new_finnhub, new_gdelt)
                
                print("\n" + "🎉" * 35)
                print("✅ 任务完成！")
                print("🎉" * 35)
                print(f"\n📂 数据文件: {saved_path}")
                print(f"📤 请将JSON文件发送给队友6进行合并\n")
            else:
                print("\n⚠️  没有新数据可保存")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断采集")
            print("提示: 下次运行将自动从中断处继续\n")
        except Exception as e:
            print(f"\n❌发生错误: {e}")
            import traceback
            traceback.print_exc()


# ============================================
# 主函数
# ============================================

def main():
    """主函数"""
    # 使用配置的teammate
    collector = IncrementalNewsCollector(CURRENT_TEAMMATE)
    collector.run()


if __name__ == "__main__":
    main()