"""
HyDE (Hypothetical Document Embeddings) 检索模块

核心思想：
    先让 LLM 根据 Query 生成若干"假设答案文档"，
    再对这些假设文档向量化，与 Query 向量融合后检索真实文档。

参考论文：Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE, EMNLP 2023)
"""

import numpy as np
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from loguru import logger

from src.generation.llm_client import LLMClient


HYDE_PROMPT_TEMPLATE = """你正在模拟知识库中的一段新闻正文摘录（约 150 字）。
要求：
1. 必须保留问题中的人名、地名、数字、专有名词；
2. 客观陈述，不要「我无法回答」类拒答；
3. 只输出正文，不要引导语。

问题：{query}

正文摘录："""


class HyDERetriever:
    """
    基于假设文档嵌入的检索器。

    工作流：
        Query
          ↓
        LLM 生成 N 份假设文档
          ↓
        embed(query) 与 mean(embed(hyp_docs)) 按 fusion_alpha 融合
          ↓
        用融合向量检索 FAISS → Top-K Documents
    """

    def __init__(
        self,
        vectorstore: FAISS,
        llm_client: LLMClient,
        top_k: int = 10,
        num_hypothetical_docs: int = 1,
        temperature: float = 0.1,
        max_tokens: int = 200,
        fusion_alpha: float = 0.6,
    ):
        self.vectorstore = vectorstore
        self.llm_client = llm_client
        self.top_k = top_k
        self.num_hypothetical_docs = num_hypothetical_docs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fusion_alpha = min(max(fusion_alpha, 0.0), 1.0)

    def _generate_hypothetical_docs(self, query: str) -> List[str]:
        """调用 LLM 生成多份假设文档。"""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        hypothetical_docs = []

        for i in range(self.num_hypothetical_docs):
            try:
                doc_text = self.llm_client.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                hypothetical_docs.append(doc_text.strip())
                logger.debug(f"[HyDE] 生成假设文档 {i+1}/{self.num_hypothetical_docs}: {doc_text[:50]}...")
            except Exception as e:
                logger.warning(f"[HyDE] 第 {i+1} 份假设文档生成失败: {e}")

        if not hypothetical_docs:
            logger.warning("[HyDE] 所有假设文档生成失败，回退到原始 Query")
            hypothetical_docs = [query]

        return hypothetical_docs

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量向量化文本，返回 (N, dim) 的 numpy 数组。"""
        embeddings = self.vectorstore.embedding_function.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return vec
        return vec / norm

    def _build_query_vector(self, query: str, hypothetical_docs: List[str]) -> List[float]:
        """Query 与假设文档 embedding 线性融合（BGE 归一化空间）。"""
        alpha = self.fusion_alpha
        if alpha >= 1.0:
            return self._embed_texts([query])[0].tolist()

        query_vec = self._embed_texts([query])[0]
        hyp_vec = self._embed_texts(hypothetical_docs).mean(axis=0)

        if alpha <= 0.0:
            fused = hyp_vec
        else:
            fused = alpha * query_vec + (1.0 - alpha) * hyp_vec

        return self._normalize(fused).tolist()

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        执行 HyDE 检索。

        Args:
            query: 用户原始问题
            top_k: 返回文档数量

        Returns:
            (Document, score) 元组列表
        """
        k = top_k or self.top_k
        logger.debug(f"[HyDE] 检索 query='{query}', top_k={k}, fusion_alpha={self.fusion_alpha}")

        hypothetical_docs = self._generate_hypothetical_docs(query)
        query_vector = self._build_query_vector(query, hypothetical_docs)

        doc_score_pairs = self.vectorstore.similarity_search_with_score_by_vector(
            query_vector, k=k
        )

        logger.debug(f"[HyDE] 检索到 {len(doc_score_pairs)} 个候选文档")
        return doc_score_pairs

    def retrieve_docs(self, query: str, top_k: int = None) -> List[Document]:
        """仅返回 Document 列表（不含 score）。"""
        return [doc for doc, _ in self.retrieve(query, top_k)]
