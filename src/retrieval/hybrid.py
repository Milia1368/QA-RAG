"""
两阶段混合检索：Naive ∪ HyDE 并集去重，保留最优距离后送入 Reranker。
"""

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from src.retrieval.adaptive_hyde import AdaptiveHyDERetriever
from src.retrieval.hyde import HyDERetriever
from src.retrieval.naive_rag import NaiveRetriever


def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    chunk = doc.metadata.get("chunk_index", "")
    return f"{source}:{chunk}"


class HybridRetriever:
    """
    Stage 1: Naive top_k +（可选 Adaptive 门控）HyDE top_k
    Merge:  按 chunk 去重，保留较小 L2 距离（更相似）
    """

    def __init__(
        self,
        naive_retriever: NaiveRetriever,
        hyde_retriever: HyDERetriever,
        merge_top_k: int = 12,
        use_adaptive_gate: bool = True,
        adaptive_retriever: Optional[AdaptiveHyDERetriever] = None,
    ):
        self.naive_retriever = naive_retriever
        self.hyde_retriever = hyde_retriever
        self.merge_top_k = merge_top_k
        self.use_adaptive_gate = use_adaptive_gate
        self.adaptive_retriever = adaptive_retriever
        self.last_used_hyde: bool = False

    @staticmethod
    def _merge_pairs(
        *pair_lists: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        best: Dict[str, Tuple[Document, float]] = {}
        for pairs in pair_lists:
            for doc, score in pairs:
                key = _doc_key(doc)
                if key not in best or score < best[key][1]:
                    best[key] = (doc, score)
        merged = sorted(best.values(), key=lambda x: x[1])
        return merged

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        k = top_k or self.merge_top_k
        naive_pairs = self.naive_retriever.retrieve(query, top_k=self.naive_retriever.top_k)

        use_hyde = True
        if self.use_adaptive_gate and self.adaptive_retriever is not None:
            naive_scores = [s for _, s in naive_pairs]
            use_hyde, reason = self.adaptive_retriever._should_use_hyde(naive_scores)
            logger.debug(f"[Hybrid] adaptive_gate={use_hyde} reason={reason}")

        self.last_used_hyde = use_hyde
        pair_lists = [naive_pairs]
        if use_hyde:
            hyde_pairs = self.hyde_retriever.retrieve(query, top_k=self.hyde_retriever.top_k)
            pair_lists.append(hyde_pairs)

        merged = self._merge_pairs(*pair_lists)
        logger.debug(
            f"[Hybrid] Naive={len(naive_pairs)} HyDE={'on' if use_hyde else 'off'} "
            f"→ merge {len(merged[:k])}/{len(merged)}"
        )
        return merged[:k]

    def retrieve_docs(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        return [doc for doc, _ in self.retrieve(query, top_k=top_k)]
