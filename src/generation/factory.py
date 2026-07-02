"""从配置装配 RAG Pipeline 组件。"""

from src.generation.llm_client import LLMClient
from src.generation.pipeline import RAGPipeline
from src.ingestion.indexer import load_index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.adaptive_hyde import AdaptiveHyDERetriever
from src.retrieval.hyde import HyDERetriever
from src.retrieval.naive_rag import NaiveRetriever
from src.retrieval.reranker import BGEReranker


def build_llm_client(cfg: dict) -> LLMClient:
    """从配置创建 LLMClient，忽略 stream 等非构造参数。"""
    llm_cfg = cfg["llm"]
    return LLMClient(
        api_base=llm_cfg["api_base"],
        api_key=llm_cfg.get("api_key", "EMPTY"),
        model_name=llm_cfg["model_name"],
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 1024),
    )


def build_pipeline(cfg: dict) -> RAGPipeline:
    embedding_cfg = cfg["embedding"]
    vectorstore = load_index(
        cfg["ingestion"]["index_path"],
        embedding_cfg["model_name"],
        embedding_cfg["device"],
    )
    llm_client = build_llm_client(cfg)
    retrieval_cfg = cfg["retrieval"]
    reranker_cfg = cfg["reranker"]
    adaptive_cfg = retrieval_cfg.get("adaptive_hyde", {})
    hyde_cfg = retrieval_cfg.get("hyde", {})
    hybrid_cfg = retrieval_cfg.get("hybrid", {})

    naive_retriever = NaiveRetriever(vectorstore, retrieval_cfg["top_k"])
    hyde_retriever = HyDERetriever(
        vectorstore,
        llm_client,
        retrieval_cfg["top_k"],
        num_hypothetical_docs=hyde_cfg.get("num_hypothetical_docs", 1),
        temperature=hyde_cfg.get("temperature", 0.1),
        max_tokens=hyde_cfg.get("max_tokens", 200),
        fusion_alpha=hyde_cfg.get("fusion_alpha", 0.6),
    )
    adaptive_hyde_retriever = AdaptiveHyDERetriever(
        naive_retriever=naive_retriever,
        hyde_retriever=hyde_retriever,
        top_k=retrieval_cfg["top_k"],
        distance_threshold=adaptive_cfg.get("distance_threshold", 0.72),
        score_gap_threshold=adaptive_cfg.get("score_gap_threshold", 0.04),
    )
    hybrid_retriever = HybridRetriever(
        naive_retriever=naive_retriever,
        hyde_retriever=hyde_retriever,
        merge_top_k=hybrid_cfg.get("merge_top_k", 12),
        use_adaptive_gate=hybrid_cfg.get("use_adaptive_gate", True),
        adaptive_retriever=adaptive_hyde_retriever,
    )

    return RAGPipeline(
        naive_retriever=naive_retriever,
        hyde_retriever=hyde_retriever,
        adaptive_hyde_retriever=adaptive_hyde_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=BGEReranker(
            reranker_cfg["model_name"],
            device=reranker_cfg["device"],
            max_length=reranker_cfg.get("max_length", 256),
            batch_size=reranker_cfg.get("batch_size", 2),
        ),
        llm_client=llm_client,
        retrieval_top_k=retrieval_cfg["top_k"],
        rerank_top_k=retrieval_cfg["rerank_top_k"],
    )
