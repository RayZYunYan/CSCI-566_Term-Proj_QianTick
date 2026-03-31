"""
News Feature Extractor 运行入口 (子任务 C)
------------------------------------------
实际逻辑在 src/news/aggregator.py。

运行方式：
    python -m src.news.feature_extractor
    python -m src.news.feature_extractor --config configs/news_config.yaml
"""

import argparse
from src.news.aggregator import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/news_config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
