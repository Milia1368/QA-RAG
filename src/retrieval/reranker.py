"""
BGE Reranker 精排模块
对粗排 Top-K 候选文档进行交叉编码器重排序，提升最终召回精度。
"""

from typing import List, Tuple

import torch
from langchain_core.documents import Document
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    """
    基于 BAAI/bge-reranker-large 的交叉编码器重排序器。

    与双塔 Embedding 不同，Reranker 将 (query, doc) 拼接后联合编码，
    能捕捉更细粒度的语义交互，精度显著高于向量相似度。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "mps",
        max_length: int = 256,
        batch_size: int = 2,
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)

        logger.info(f"加载 Reranker 模型: {model_name} (batch_size={self.batch_size})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        logger.info("Reranker 模型加载完成")

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        对候选文档列表进行精排。

        Args:
            query: 用户问题
            docs: 粗排候选文档列表
            top_k: 精排后保留数量

        Returns:
            按 rerank score 降序排列的 (Document, score) 列表
        """
        if not docs:
            return []

        all_scores: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch_docs = docs[start : start + self.batch_size]
            pairs = [[query, doc.page_content] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            batch_scores = self.model(**inputs).logits.squeeze(-1)
            if batch_scores.dim() == 0:
                batch_scores = batch_scores.unsqueeze(0)
            all_scores.extend(torch.sigmoid(batch_scores).cpu().tolist())

            del inputs, batch_scores
            if self.device == "mps":
                torch.mps.empty_cache()

        ranked = sorted(zip(docs, all_scores), key=lambda x: x[1], reverse=True)
        result = ranked[:top_k]

        logger.debug(
            f"[Reranker] {len(docs)} 候选 → top-{top_k}，"
            f"最高分: {result[0][1]:.4f}"
        )
        return result

    def rerank_docs(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """仅返回重排后的 Document 列表（不含 score）。"""
        return [doc for doc, _ in self.rerank(query, docs, top_k)]
