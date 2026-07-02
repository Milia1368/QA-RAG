"""
评估指标模块
实现 Faithfulness（忠实度）和 MRR@K（Mean Reciprocal Rank）两个核心指标。

Faithfulness：衡量答案中有多少声明可被检索到的上下文支持，用于量化幻觉率。
MRR@K：衡量检索系统在前 K 个结果中第一个相关文档的平均排名倒数。
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.generation.llm_client import LLMClient


def extract_doc_id(metadata: dict) -> str:
    """从 chunk metadata 提取文档 ID（与 eval_dataset 中的 doc_id 对齐）。"""
    if doc_id := metadata.get("doc_id"):
        return doc_id
    source = metadata.get("source", "")
    return Path(source).stem if source else ""


def extract_doc_ids_from_results(docs: list) -> List[str]:
    """从检索结果中去重提取 doc_id 列表，保持排名顺序。"""
    seen = set()
    doc_ids = []
    for doc in docs:
        doc_id = extract_doc_id(doc.metadata)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def normalize_answer(text: str) -> str:
    """简单归一化，便于答案匹配。"""
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("《", "").replace("》", "")
    return text


def compute_answer_accuracy(
    predicted: str,
    reference: str,
) -> float:
    """
    计算答案准确率（0/1）。

    规则：归一化后 reference 出现在 predicted 中即视为正确。
    参考答案为「无法推断」时，predicted 含「无法」或「推断」也视为正确。
    """
    if not reference:
        return 0.0

    pred_norm = normalize_answer(predicted)
    ref_norm = normalize_answer(reference)

    if ref_norm == "无法推断":
        if any(kw in predicted for kw in ("无法", "不能", "未提及", "没有提到", "推断")):
            return 1.0
        return 0.0

    if not pred_norm:
        return 0.0

    return 1.0 if ref_norm in pred_norm or pred_norm in ref_norm else 0.0


# ── Faithfulness ──────────────────────────────────────────────────────────────

FAITHFULNESS_SYSTEM = """你是一位严格的信息核查员。
你的任务是判断答案中的每个陈述是否可以从给定的上下文文档中得到直接支持。
对于每个陈述，回答"支持"或"不支持"。不要作额外解释。"""

FAITHFULNESS_USER_TEMPLATE = """上下文文档：
{context}

待核查答案：
{answer}

请逐句分析答案中的陈述，并判断每句话是否能从上下文中找到支持。
最后给出以下格式的 JSON：
{{"supported_count": <支持的陈述数>, "total_count": <总陈述数>, "faithfulness": <比率，保留两位小数>}}"""


def compute_faithfulness(
    answer: str,
    context_docs: list,
    llm_client: LLMClient,
) -> float:
    """
    用 LLM 作为评判者（LLM-as-Judge）计算 Faithfulness 分数。

    原理：让 LLM 逐句检查答案的每个陈述是否有上下文支持，
    Faithfulness = 有支持的陈述数 / 总陈述数

    Args:
        answer: 模型生成的答案
        context_docs: 检索到的文档列表
        llm_client: LLM 客户端

    Returns:
        0.0 ~ 1.0 之间的忠实度分数
    """
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = FAITHFULNESS_USER_TEMPLATE.format(context=context, answer=answer)

    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt=FAITHFULNESS_SYSTEM,
            temperature=0.0,
        )
        # 提取 JSON 中的 faithfulness 值
        match = re.search(r'"faithfulness"\s*:\s*([\d.]+)', response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"[Metrics] Faithfulness 计算失败: {e}")

    return 0.0


# ── MRR@K ──────────────────────────────────────────────────────────────────────

def compute_mrr_at_k(
    retrieved_doc_ids: List[List[str]],
    relevant_doc_ids: List[List[str]],
    k: int = 5,
) -> float:
    """
    计算 MRR@K（Mean Reciprocal Rank at K）。

    Args:
        retrieved_doc_ids: 每个 query 的检索结果文档 ID 列表（按排名排序）
        relevant_doc_ids:  每个 query 的真实相关文档 ID 列表
        k: 只考虑前 K 个检索结果

    Returns:
        MRR@K 分数（0.0 ~ 1.0）

    Example:
        >>> retrieved = [["doc3", "doc1", "doc2"], ["doc2", "doc3"]]
        >>> relevant  = [["doc1", "doc2"], ["doc2"]]
        >>> compute_mrr_at_k(retrieved, relevant, k=5)
        0.75  # query1: 1/2=0.5, query2: 1/1=1.0 → (0.5+1.0)/2=0.75
    """
    assert len(retrieved_doc_ids) == len(relevant_doc_ids), \
        "retrieved 与 relevant 列表长度必须相同"

    reciprocal_ranks = []
    for retrieved, relevant_set in zip(retrieved_doc_ids, relevant_doc_ids):
        relevant_set = set(relevant_set)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return round(mrr, 4)


def compute_recall_at_k(
    retrieved_doc_ids: List[List[str]],
    relevant_doc_ids: List[List[str]],
    k: int = 5,
) -> float:
    """计算 Recall@K，作为 MRR@K 的补充指标。"""
    recalls = []
    for retrieved, relevant in zip(retrieved_doc_ids, relevant_doc_ids):
        relevant_set = set(relevant)
        hits = len(relevant_set & set(retrieved[:k]))
        recalls.append(hits / len(relevant_set) if relevant_set else 0.0)
    return round(sum(recalls) / len(recalls), 4) if recalls else 0.0
