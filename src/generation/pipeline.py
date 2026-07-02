"""
RAG Pipeline 主流程
整合检索（Naive / HyDE）→ Rerank → 生成，对外暴露统一接口。
"""

from dataclasses import dataclass, field
from typing import Generator, List, Literal

from langchain_core.documents import Document
from loguru import logger

from src.generation.llm_client import LLMClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.adaptive_hyde import AdaptiveHyDERetriever
from src.retrieval.naive_rag import NaiveRetriever
from src.retrieval.hyde import HyDERetriever
from src.retrieval.reranker import BGEReranker

RetrievalMode = Literal["naive", "hyde", "adaptive_hyde", "hybrid"]


@dataclass
class RAGResult:
    """RAG 问答结果，包含答案与溯源信息。"""
    answer: str
    source_docs: List[Document]
    retrieval_mode: str
    query: str
    sources: List[dict] = field(default_factory=list)

    def __post_init__(self):
        """从 source_docs 提取来源摘要，便于前端展示。"""
        self.sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "snippet": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
            for doc in self.source_docs
        ]


class RAGPipeline:
    """
    两阶段 RAG Pipeline：
        粗排（NaiveRAG / HyDE）→ BGE Reranker 精排 → Qwen 生成答案

    支持：
        - mode="naive": 标准向量检索
        - mode="hyde":  HyDE 假设文档增强检索
        - mode="adaptive_hyde": 置信度门控，按需启用 HyDE
        - mode="hybrid": Naive ∪ HyDE 两阶段并集检索
        - 流式 / 非流式输出
        - 答案溯源（source_docs 字段）
    """

    def __init__(
        self,
        naive_retriever: NaiveRetriever,
        hyde_retriever: HyDERetriever,
        reranker: BGEReranker,
        llm_client: LLMClient,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
        adaptive_hyde_retriever: AdaptiveHyDERetriever = None,
        hybrid_retriever: HybridRetriever = None,
    ):
        self.naive_retriever = naive_retriever
        self.hyde_retriever = hyde_retriever
        self.adaptive_hyde_retriever = adaptive_hyde_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def _retrieve(
        self,
        query: str,
        mode: RetrievalMode = "hyde",
    ) -> List[Document]:
        """Step 1: 粗排检索。"""
        if mode == "hyde":
            docs = self.hyde_retriever.retrieve_docs(query, top_k=self.retrieval_top_k)
        elif mode == "adaptive_hyde":
            if self.adaptive_hyde_retriever is None:
                raise ValueError("adaptive_hyde 模式未配置 AdaptiveHyDERetriever")
            docs = self.adaptive_hyde_retriever.retrieve_docs(query, top_k=self.retrieval_top_k)
        elif mode == "hybrid":
            if self.hybrid_retriever is None:
                raise ValueError("hybrid 模式未配置 HybridRetriever")
            docs = self.hybrid_retriever.retrieve_docs(
                query, top_k=self.retrieval_top_k
            )
        else:
            docs = self.naive_retriever.retrieve_docs(query, top_k=self.retrieval_top_k)
        logger.debug(f"[Pipeline] 粗排召回 {len(docs)} 篇，mode={mode}")
        return docs

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Step 2: Reranker 精排。"""
        reranked = self.reranker.rerank_docs(query, docs, top_k=self.rerank_top_k)
        logger.debug(f"[Pipeline] 精排后保留 {len(reranked)} 篇")
        return reranked

    def run(
        self,
        query: str,
        mode: RetrievalMode = "hyde",
    ) -> RAGResult:
        """
        执行完整 RAG Pipeline，返回答案与来源（非流式）。

        Args:
            query: 用户问题
            mode: 检索策略

        Returns:
            RAGResult 对象
        """
        logger.info(f"[Pipeline] 开始处理: query='{query}', mode={mode}")

        # Step 1: 检索
        candidates = self._retrieve(query, mode)

        # Step 2: Rerank
        source_docs = self._rerank(query, candidates)

        # Step 3: 构造 Prompt
        system_prompt, user_prompt = build_rag_prompt(query, source_docs)

        # Step 4: 生成答案
        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            stream=False,
        )

        logger.info(f"[Pipeline] 生成完成，答案长度: {len(answer)} 字")

        return RAGResult(
            answer=answer,
            source_docs=source_docs,
            retrieval_mode=mode,
            query=query,
        )

    def stream_run(
        self,
        query: str,
        mode: RetrievalMode = "hyde",
    ) -> Generator[str, None, None]:
        """
        流式执行 RAG Pipeline，逐 token yield 输出。
        source_docs 信息通过最后一个 SSE 事件发送（见 api/routers.py）。

        Args:
            query: 用户问题
            mode: 检索策略

        Yields:
            答案文本 delta
        """
        logger.info(f"[Pipeline] 流式处理: query='{query}', mode={mode}")

        candidates = self._retrieve(query, mode)
        source_docs = self._rerank(query, candidates)
        system_prompt, user_prompt = build_rag_prompt(query, source_docs)

        # 先 yield 来源信息（特殊前缀，前端解析时过滤）
        import json
        sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "snippet": doc.page_content[:150],
            }
            for doc in source_docs
        ]
        yield f"[SOURCES]{json.dumps(sources, ensure_ascii=False)}[/SOURCES]\n"

        # 流式 yield 答案
        for token in self.llm_client.stream_generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        ):
            yield token
