"""评估指标合理性测试（无需 GPU）。"""

from langchain_core.documents import Document

from src.evaluation.metrics import (
    compute_mrr_at_k,
    compute_recall_at_k,
    extract_doc_ids_from_results,
)


def test_mrr_uses_coarse_ranking_not_reranked_subset():
    """MRR@5 应基于粗排 doc 列表；rerank 后若只剩 3 条会低估排名。"""
    coarse_docs = [
        Document(page_content="a", metadata={"doc_id": "wrong1"}),
        Document(page_content="b", metadata={"doc_id": "target"}),
        Document(page_content="c", metadata={"doc_id": "wrong2"}),
        Document(page_content="d", metadata={"doc_id": "wrong3"}),
        Document(page_content="e", metadata={"doc_id": "wrong4"}),
    ]
    reranked_docs = coarse_docs[:1]  # rerank 后 target 被截掉

    coarse_ids = extract_doc_ids_from_results(coarse_docs)
    reranked_ids = extract_doc_ids_from_results(reranked_docs)

    relevant = [["target"]]
    mrr_coarse = compute_mrr_at_k([coarse_ids], relevant, k=5)
    mrr_reranked = compute_mrr_at_k([reranked_ids], relevant, k=5)

    assert mrr_coarse == 0.5  # rank 2
    assert mrr_reranked == 0.0  # target 不在 rerank 结果中


def test_recall_at_k_single_doc_qa():
    """QuestAnswer1Doc 每题只有一个相关 doc，Recall@5 等价于 Hit@5。"""
    retrieved = [["a", "b", "target", "c", "d"]]
    relevant = [["target"]]
    assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0
    assert compute_mrr_at_k(retrieved, relevant, k=5) == 0.3333
