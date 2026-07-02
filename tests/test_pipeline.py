"""
RAG Pipeline 集成测试
使用 mock 替换真实模型，验证 Pipeline 的完整流程与边界处理。
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.generation.pipeline import RAGPipeline, RAGResult


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="差旅报销需提交发票原件及审批表。",
            metadata={"source": "hr.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="年假申请需提前三个工作日提交。",
            metadata={"source": "hr.pdf", "chunk_index": 1},
        ),
    ]


@pytest.fixture
def mock_pipeline(sample_docs):
    """构造完全 mock 的 RAGPipeline，不依赖真实 GPU 或模型。"""
    naive_retriever = MagicMock()
    naive_retriever.retrieve_docs.return_value = sample_docs

    hyde_retriever = MagicMock()
    hyde_retriever.retrieve_docs.return_value = sample_docs

    adaptive_hyde_retriever = MagicMock()
    adaptive_hyde_retriever.retrieve_docs.return_value = sample_docs

    hybrid_retriever = MagicMock()
    hybrid_retriever.retrieve_docs.return_value = sample_docs

    reranker = MagicMock()
    reranker.rerank_docs.return_value = sample_docs[:1]

    llm_client = MagicMock()
    llm_client.generate.return_value = "差旅报销需提交发票原件及审批表，并经主管审批。"
    llm_client.stream_generate.return_value = iter(["差旅", "报销", "需提交", "发票。"])

    return RAGPipeline(
        naive_retriever=naive_retriever,
        hyde_retriever=hyde_retriever,
        adaptive_hyde_retriever=adaptive_hyde_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_client=llm_client,
    )


class TestRAGPipeline:
    def test_run_returns_rag_result(self, mock_pipeline):
        result = mock_pipeline.run("差旅报销流程是什么？", mode="naive")
        assert isinstance(result, RAGResult)
        assert result.answer != ""
        assert len(result.source_docs) > 0

    def test_run_naive_mode_calls_naive_retriever(self, mock_pipeline):
        mock_pipeline.run("测试问题", mode="naive")
        mock_pipeline.naive_retriever.retrieve_docs.assert_called_once()
        mock_pipeline.hyde_retriever.retrieve_docs.assert_not_called()

    def test_run_hyde_mode_calls_hyde_retriever(self, mock_pipeline):
        mock_pipeline.run("测试问题", mode="hyde")
        mock_pipeline.hyde_retriever.retrieve_docs.assert_called_once()
        mock_pipeline.naive_retriever.retrieve_docs.assert_not_called()

    def test_run_adaptive_hyde_mode_calls_adaptive_retriever(self, mock_pipeline):
        mock_pipeline.run("测试问题", mode="adaptive_hyde")
        mock_pipeline.adaptive_hyde_retriever.retrieve_docs.assert_called_once()
        mock_pipeline.naive_retriever.retrieve_docs.assert_not_called()
        mock_pipeline.hyde_retriever.retrieve_docs.assert_not_called()

    def test_run_hybrid_mode_calls_hybrid_retriever(self, mock_pipeline):
        mock_pipeline.run("测试问题", mode="hybrid")
        mock_pipeline.hybrid_retriever.retrieve_docs.assert_called_once()
        mock_pipeline.naive_retriever.retrieve_docs.assert_not_called()
        mock_pipeline.hyde_retriever.retrieve_docs.assert_not_called()

    def test_run_includes_sources(self, mock_pipeline):
        result = mock_pipeline.run("差旅报销流程？", mode="naive")
        assert len(result.sources) > 0
        assert "source" in result.sources[0]
        assert "snippet" in result.sources[0]

    def test_stream_run_yields_tokens(self, mock_pipeline):
        chunks = list(mock_pipeline.stream_run("差旅报销流程？", mode="naive"))
        # 第一个 chunk 是 SOURCES 信息帧
        assert chunks[0].startswith("[SOURCES]")
        # 后续有 token 内容
        assert len(chunks) > 1

    def test_rag_result_sources_populated(self, sample_docs):
        """RAGResult 的 __post_init__ 应自动填充 sources 列表。"""
        result = RAGResult(
            answer="测试答案",
            source_docs=sample_docs,
            retrieval_mode="naive",
            query="测试",
        )
        assert len(result.sources) == len(sample_docs)
        assert result.sources[0]["source"] == "hr.pdf"
