"""
FinBERT Encoder (子任务 B)
--------------------------
输入：headline 文本字符串列表
输出：Tensor (N, 771)，前 768 维为 [CLS] embedding，后 3 维为 sentiment 概率

负责人：子任务 B
对应文件：src/news/encoder.py
输出文件：data/features/news_article_embeddings.parquet
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


FINBERT_MODEL = "ProsusAI/finbert"


class FinBERTEncoder(nn.Module):
    """
    对批量新闻文本输出 771 维特征向量（768-dim [CLS] + 3-dim sentiment 拼接）。
    FinBERT 权重冻结，仅作特征提取。

    TODO (子任务 B)：
        1. 实现 encode()：
               tokenize（max_length=512, truncation=True）
               → bert.forward()
               → 提取 hidden_states[:, 0, :] 作为 [CLS]（768 维）
               → 提取 logits 经 softmax 作为 sentiment（3 维）
               → torch.cat([cls_emb, sentiment], dim=-1) 得到 (N, 771)
        2. 实现 encode_and_save()：
               批量编码 news_cleaned.parquet 中的所有文章
               断点续跑：每编码 checkpoint_every 篇追加保存到临时文件
               重启时跳过已编码的 article_id
               最终输出 news_article_embeddings.parquet

    Mock 接口（供 C 开发时使用，无需加载真实模型）：
        mock_output = torch.randn(N, 771)
        mock_output[:, 768:] = torch.softmax(mock_output[:, 768:], dim=-1)
    """

    def __init__(self, model_name: str = FINBERT_MODEL, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.to(self.device)
        self.bert.eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: 长度为 N 的字符串列表

        Returns:
            Tensor (N, 771)：前 768 维 [CLS]，后 3 维 sentiment 概率
        """
        # TODO: 实现编码逻辑
        raise NotImplementedError

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

        TODO (子任务 B)：
            - 读取 cleaned_parquet，检查 output_parquet 已存在的 article_id（断点续跑）
            - 分批调用 encode()，每 checkpoint_every 篇追加写入 output_parquet
        """
        # TODO: 实现断点续跑批量编码
        raise NotImplementedError
