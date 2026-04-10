"""
FinBERT Encoder (子任务 B)
--------------------------
输入：headline 文本字符串列表
输出：Tensor (N, 771)，前 768 维为 [CLS] embedding，后 3 维为 sentiment 概率

负责人：子任务 B
对应文件：src/news/encoder.py
输出文件：data/features/news_article_embeddings.parquet
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
import numpy as np

FINBERT_MODEL = "ProsusAI/finbert"


class FinBERTEncoder(nn.Module):
    """
    对批量新闻文本输出 771 维特征向量（768-dim [CLS] + 3-dim sentiment 拼接）。
    FinBERT 权重冻结，仅作特征提取。

    Mock 接口（供 C 开发时使用，无需加载真实模型）：
        mock_output = torch.randn(N, 771)
        mock_output[:, 768:] = torch.softmax(mock_output[:, 768:], dim=-1)
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.to(self.device)
        self.bert.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty((0, 771), dtype=torch.float32)

        # tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bert(**inputs, output_hidden_states=True)

            # CLS embedding (batch, 768)
            cls_emb = outputs.hidden_states[-1][:, 0, :]

            # raw sentiment logits (batch, 3) — negative, neutral, positive
            sentiment = outputs.logits

            # concat -> (batch, 771)
            features = torch.cat([cls_emb, sentiment], dim=-1)

        return features.to(torch.float32)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts)

    def encode_and_save(
        self,
        cleaned_parquet: str,
        output_parquet: str,
        batch_size: int = 64,
        checkpoint_every: int = 10000,
    ) -> None:
        """
        批量编码 news_cleaned.parquet，断点续跑，输出 news_article_embeddings.parquet。

        输出 schema：
            ticker (str), date (str YYYY-MM-DD), article_id (str),
            feat_0 ~ feat_770 (float)
        """
        cleaned_parquet = Path(cleaned_parquet)
        output_parquet = Path(output_parquet)

        if not cleaned_parquet.exists():
            raise FileNotFoundError(f"Input file not found: {cleaned_parquet}")

        output_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(cleaned_parquet)

        required_cols = ["ticker", "date", "article_id", "headline"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = df["date"].astype(str)
        df["article_id"] = df["article_id"].astype(str)
        df["headline"] = df["headline"].fillna("").astype(str)

        df = df.drop_duplicates(subset=["article_id", "date"]).reset_index(drop=True)

        existing_pairs = set()
        if output_parquet.exists():
            old_df = pd.read_parquet(output_parquet, columns=["article_id", "date"])
            existing_pairs = set(
                zip(old_df["article_id"].astype(str), old_df["date"].astype(str))
            )

        if existing_pairs:
            mask = [
                (str(a), str(d)) not in existing_pairs
                for a, d in zip(df["article_id"], df["date"])
            ]
            df = df[mask].reset_index(drop=True)

        if df.empty:
            print("No new articles to encode. Output is already up to date.")
            return

        feature_cols = [f"feat_{i}" for i in range(771)]
        buffer_rows = []
        processed_since_last_save = 0
        total = len(df)

        def flush_buffer(rows: list[dict]) -> None:
            if not rows:
                return

            chunk_df = pd.DataFrame(rows)
            ordered_cols = ["ticker", "date", "article_id"] + feature_cols
            chunk_df = chunk_df[ordered_cols]

            if output_parquet.exists():
                existing_df = pd.read_parquet(output_parquet)
                merged_df = pd.concat([existing_df, chunk_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=["article_id", "date"], keep="last")
                merged_df.to_parquet(output_parquet, index=False)
            else:
                chunk_df.to_parquet(output_parquet, index=False)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_df = df.iloc[start:end]

            texts = batch_df["headline"].tolist()
            features = self.encode(texts)  # (B, 771)
            features_np = features.detach().cpu().numpy().astype(np.float32)

            meta = batch_df[["ticker", "date", "article_id"]].reset_index(drop=True)

            for i in range(len(batch_df)):
                row_dict = {
                    "ticker": meta.loc[i, "ticker"],
                    "date": meta.loc[i, "date"],
                    "article_id": meta.loc[i, "article_id"],
                }
                for j in range(771):
                    row_dict[f"feat_{j}"] = float(features_np[i, j])
                buffer_rows.append(row_dict)

            processed_since_last_save += len(batch_df)

            if processed_since_last_save >= checkpoint_every:
                flush_buffer(buffer_rows)
                print(f"Checkpoint saved: {end}/{total}")
                buffer_rows = []
                processed_since_last_save = 0

        if buffer_rows:
            flush_buffer(buffer_rows)

        print(f"Encoding finished. Saved {total} new articles to {output_parquet}")
