"""Adaptive HyDE 路由逻辑单元测试（无需 GPU）。"""

from unittest.mock import MagicMock

import pytest

from src.retrieval.adaptive_hyde import AdaptiveHyDERetriever


@pytest.fixture
def adaptive_retriever():
    naive = MagicMock()
    hyde = MagicMock()
    hyde.retrieve_docs.return_value = ["hyde_doc"]
    return AdaptiveHyDERetriever(
        naive_retriever=naive,
        hyde_retriever=hyde,
        distance_threshold=0.45,
        score_gap_threshold=0.08,
    ), naive, hyde


class TestAdaptiveHyDERouting:
    def test_high_confidence_uses_naive(self, adaptive_retriever):
        retriever, naive, hyde = adaptive_retriever
        naive.retrieve.return_value = [("doc_a", 0.2), ("doc_b", 0.5)]

        docs, route = retriever.retrieve("测试问题")

        assert route == "naive"
        assert docs == ["doc_a"]
        hyde.retrieve_docs.assert_not_called()

    def test_low_confidence_triggers_hyde(self, adaptive_retriever):
        retriever, naive, hyde = adaptive_retriever
        naive.retrieve.return_value = [("doc_a", 0.6), ("doc_b", 0.7)]

        docs, route = retriever.retrieve("测试问题")

        assert route == "hyde"
        assert docs == ["hyde_doc"]
        hyde.retrieve_docs.assert_called_once()

    def test_ambiguous_top2_triggers_hyde(self, adaptive_retriever):
        retriever, naive, hyde = adaptive_retriever
        naive.retrieve.return_value = [("doc_a", 0.3), ("doc_b", 0.34)]

        docs, route = retriever.retrieve("测试问题")

        assert route == "hyde"
        hyde.retrieve_docs.assert_called_once()

    def test_empty_naive_triggers_hyde(self, adaptive_retriever):
        retriever, naive, hyde = adaptive_retriever
        naive.retrieve.return_value = []

        _, route = retriever.retrieve("测试问题")

        assert route == "hyde"
        hyde.retrieve_docs.assert_called_once()
