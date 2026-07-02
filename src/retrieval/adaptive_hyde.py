"""
Adaptive HyDE 检索模块

策略（置信度门控）：
    1. 先执行 Naive 向量检索，获取 top-1/top-2 的 L2 距离
    2. 若 top-1 距离足够小（高置信命中）且 top-1 与 top-2 差距明显 → 直接用 Naive 结果
    3. 否则（低置信或 top 结果歧义）→ 回退到 HyDE 检索

适用场景：
    - 简单 factual 问题：省 LLM 调用，降低延迟与内存压力
    - 歧义/语义跨度大的问题：自动启用 HyDE，保留其 MRR 优势

参考：HyDE (EMNLP 2023) 的工程化自适应变体，非独立论文方法。
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from src.retrieval.hyde import HyDERetriever
from src.retrieval.naive_rag import NaiveRetriever


class AdaptiveHyDERetriever:
    """
    置信度门控的自适应 HyDE 检索器。

    FAISS + 归一化 embedding 返回 L2 距离，越小表示越相似。
    """

    def __init__(
        self,
        naive_retriever: NaiveRetriever,
        hyde_retriever: HyDERetriever,
        top_k: int = 10,
        distance_threshold: float = 0.45,
        score_gap_threshold: float = 0.08,
    ):
        self.naive_retriever = naive_retriever
        self.hyde_retriever = hyde_retriever
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.score_gap_threshold = score_gap_threshold
        self.last_route: str = "naive"
        self.last_route_reason: str = ""

    def _should_use_hyde(self, naive_scores: List[float]) -> Tuple[bool, str]:
        """
        判断是否应触发 HyDE。

        Returns:
            (use_hyde, reason)
        """
        if not naive_scores:
            return True, "empty_results"

        top1 = naive_scores[0]
        if top1 > self.distance_threshold:
            return True, "low_confidence"

        if len(naive_scores) >= 2:
            gap = naive_scores[1] - top1
            if gap < self.score_gap_threshold:
                return True, "ambiguous_top2"

        return False, "high_confidence"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Document], str]:
        """
        执行 Adaptive HyDE 检索。

        Returns:
            (documents, route)  route 为 "naive" 或 "hyde"
        """
        k = top_k or self.top_k
        naive_pairs = self.naive_retriever.retrieve(query, top_k=k)
        naive_scores = [score for _, score in naive_pairs]

        use_hyde, reason = self._should_use_hyde(naive_scores)
        self.last_route_reason = reason

        if use_hyde:
            docs = self.hyde_retriever.retrieve_docs(query, top_k=k)
            self.last_route = "hyde"
            msg = f"[AdaptiveHyDE] 触发 HyDE (reason={reason})"
            if naive_scores:
                msg += f", top1_dist={naive_scores[0]:.4f}"
            logger.debug(msg)
        else:
            docs = [doc for doc, _ in naive_pairs]
            self.last_route = "naive"
            logger.debug(
                f"[AdaptiveHyDE] 使用 Naive (reason={reason}), top1_dist={naive_scores[0]:.4f}"
            )

        return docs, self.last_route

    def retrieve_docs(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        docs, _ = self.retrieve(query, top_k=top_k)
        return docs
