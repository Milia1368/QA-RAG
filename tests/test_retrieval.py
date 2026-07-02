"""
检索模块单元测试
测试 NaiveRetriever 与 HyDERetriever 的基本行为，
以及 MRR@K 指标计算的正确性。
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from src.evaluation.metrics import compute_mrr_at_k, compute_recall_at_k
from src.retrieval.naive_rag import NaiveRetriever


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="差旅报销需提交发票原件及审批表，金额超过5000元需总监签字。",
            metadata={"source": "hr_policy.pdf", "chunk_index": 0},
        ),
        Document(
            page_content="研发规范要求所有代码必须经过 Code Review，PR 合并需至少两位同事审批。",
            metadata={"source": "dev_guideline.md", "chunk_index": 3},
        ),
        Document(
            page_content="年假申请需提前三个工作日在 OA 系统提交，部门主管审批后生效。",
            metadata={"source": "hr_policy.pdf", "chunk_index": 5},
        ),
    ]


@pytest.fixture
def mock_vectorstore(sample_docs):
    """构造一个返回固定结果的 mock FAISS vectorstore。"""
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = [
        (sample_docs[0], 0.92),
        (sample_docs[1], 0.75),
        (sample_docs[2], 0.61),
    ]
    return vs


# ── NaiveRetriever 测试 ──────────────────────────────────────────────────────

class TestNaiveRetriever:
    def test_retrieve_returns_correct_count(self, mock_vectorstore):
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, top_k=3)
        results = retriever.retrieve("差旅报销流程")
        assert len(results) == 3

    def test_retrieve_returns_doc_and_score_tuples(self, mock_vectorstore):
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, top_k=3)
        results = retriever.retrieve("差旅报销流程")
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_retrieve_docs_strips_scores(self, mock_vectorstore):
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, top_k=3)
        docs = retriever.retrieve_docs("差旅报销流程")
        assert all(isinstance(d, Document) for d in docs)

    def test_top_k_override(self, mock_vectorstore):
        """retrieve 时传入 top_k 应覆盖初始化值。"""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, top_k=10)
        retriever.retrieve("测试", top_k=2)
        mock_vectorstore.similarity_search_with_score.assert_called_with("测试", k=2)


# ── MRR@K 指标测试 ──────────────────────────────────────────────────────────

class TestMRRMetric:
    def test_perfect_retrieval(self):
        """第一个结果就是相关文档，MRR@5 = 1.0。"""
        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [["doc1"]]
        assert compute_mrr_at_k(retrieved, relevant, k=5) == 1.0

    def test_second_position(self):
        """相关文档在第二位，MRR@5 = 0.5。"""
        retrieved = [["doc2", "doc1", "doc3"]]
        relevant = [["doc1"]]
        assert compute_mrr_at_k(retrieved, relevant, k=5) == 0.5

    def test_not_retrieved(self):
        """相关文档不在 Top-K 中，MRR@K = 0.0。"""
        retrieved = [["doc2", "doc3", "doc4"]]
        relevant = [["doc1"]]
        assert compute_mrr_at_k(retrieved, relevant, k=3) == 0.0

    def test_multi_query_average(self):
        """多个 query 的 MRR 取平均值。"""
        retrieved = [
            ["doc3", "doc1", "doc2"],  # 相关在第2位 → 1/2
            ["doc2", "doc3", "doc1"],  # 相关在第1位 → 1/1
        ]
        relevant = [["doc1"], ["doc2"]]
        mrr = compute_mrr_at_k(retrieved, relevant, k=5)
        assert abs(mrr - 0.75) < 1e-6

    def test_recall_at_k(self):
        retrieved = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevant = [["doc1", "doc3", "doc6"]]
        recall = compute_recall_at_k(retrieved, relevant, k=5)
        # 命中 doc1, doc3，共2/3
        assert abs(recall - 2 / 3) < 1e-4
