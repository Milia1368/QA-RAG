"""
Naive RAG 检索模块
直接将用户 Query 向量化后进行相似度检索，不做任何 Query 变换。
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from loguru import logger


class NaiveRetriever:
    """
    标准向量检索器。

    工作流：
        Query → Embedding → FAISS ANN 搜索 → Top-K Documents
    """

    def __init__(self, vectorstore: FAISS, top_k: int = 10):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        检索与 query 最相关的文档片段。

        Args:
            query: 用户原始问题
            top_k: 返回数量，默认使用初始化时的值

        Returns:
            (Document, score) 元组列表，score 越高越相关
        """
        k = top_k or self.top_k
        logger.debug(f"[NaiveRAG] 检索 query='{query}', top_k={k}")

        doc_score_pairs = self.vectorstore.similarity_search_with_score(query, k=k)

        # FAISS 返回的是 L2 距离，转成相似度（越小越相似）
        # 对于余弦相似度索引，score 直接表示相似度
        logger.debug(f"[NaiveRAG] 检索到 {len(doc_score_pairs)} 个候选文档")
        return doc_score_pairs

    def retrieve_docs(self, query: str, top_k: int = None) -> List[Document]:
        """仅返回 Document 列表（不含 score）。"""
        return [doc for doc, _ in self.retrieve(query, top_k)]
