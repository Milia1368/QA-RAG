"""
Naive RAG vs HyDE vs Adaptive HyDE 对比实验脚本
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from loguru import logger

from src.evaluation.metrics import (
    compute_answer_accuracy,
    compute_faithfulness,
    compute_mrr_at_k,
    compute_recall_at_k,
    extract_doc_ids_from_results,
)
from src.generation.factory import build_pipeline
from src.generation.pipeline import RAGPipeline
from src.generation.prompt import build_rag_prompt

DEFAULT_EVAL_MODES = ("naive", "hyde", "adaptive_hyde", "hybrid")
EVAL_MODES = DEFAULT_EVAL_MODES
DETAIL_LIMIT = 20  # 写入报告时每模式最多保留的明细条数


@dataclass
class EvalSample:
    """评估样本：一个问题 + 标准相关文档 ID 列表"""
    query: str
    relevant_doc_ids: List[str]
    reference_answer: str = ""
    doc_id: str = ""


@dataclass
class EvalResult:
    """单个 query 的评估结果"""
    query: str
    mode: str
    retrieved_doc_ids: List[str]
    answer: str
    faithfulness: float
    answer_accuracy: float
    latency_ms: float


@dataclass
class EvalReport:
    """整体评估报告"""
    mode: str
    mrr_at_5: float
    recall_at_5: float
    avg_faithfulness: float
    avg_answer_accuracy: float
    avg_latency_ms: float
    num_samples: int
    hyde_trigger_rate: Optional[float] = None  # adaptive_hyde 模式下 HyDE 触发比例
    details: List[dict] = field(default_factory=list)


def load_eval_dataset(path: str, limit: Optional[int] = None) -> List[EvalSample]:
    """
    加载评估数据集（JSONL 格式）。

    每行格式：
        {"query": "...", "relevant_doc_ids": ["doc1", ...], "reference_answer": "..."}
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(EvalSample(**data))
            if limit is not None and len(samples) >= limit:
                break
    logger.info(f"加载评估数据集: {path}，共 {len(samples)} 条")
    return samples


def _retrieve_candidates(pipeline: RAGPipeline, query: str, mode: str):
    """按模式执行粗排检索，返回 (candidates, route)。"""
    route = None
    if mode == "hyde":
        candidates = pipeline.hyde_retriever.retrieve_docs(
            query, top_k=pipeline.retrieval_top_k
        )
    elif mode == "adaptive_hyde":
        candidates = pipeline.adaptive_hyde_retriever.retrieve_docs(
            query, top_k=pipeline.retrieval_top_k
        )
        route = pipeline.adaptive_hyde_retriever.last_route
    elif mode == "hybrid":
        candidates = pipeline.hybrid_retriever.retrieve_docs(
            query, top_k=pipeline.retrieval_top_k
        )
        route = "hyde" if pipeline.hybrid_retriever.last_used_hyde else "naive"
    else:
        candidates = pipeline.naive_retriever.retrieve_docs(
            query, top_k=pipeline.retrieval_top_k
        )
    return candidates, route


def run_eval(
    pipeline: RAGPipeline,
    samples: List[EvalSample],
    mode: str,
    retrieval_only: bool = False,
    gc_interval: int = 20,
    device: str = "mps",
) -> EvalReport:
    """
    在给定样本集上运行指定检索策略，计算评估指标。

    Args:
        pipeline: RAG Pipeline 实例
        samples: 评估样本列表
        mode: "naive" / "hyde" / "adaptive_hyde"
        retrieval_only: 仅评估检索指标，跳过 LLM 生成与 Faithfulness

    Returns:
        EvalReport 评估报告
    """
    logger.info(
        f"开始评估 mode={mode}，共 {len(samples)} 个样本"
        f"{'（仅检索）' if retrieval_only else ''}..."
    )

    all_retrieved_ids = []
    all_relevant_ids = []
    faithfulness_scores = []
    answer_accuracy_scores = []
    latencies = []
    details = []

    hyde_trigger_count = 0

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()

        # 粗排检索（MRR/Recall 在 rerank 之前评估，避免 rerank_top_k 截断导致指标失真）
        candidates, route = _retrieve_candidates(pipeline, sample.query, mode)
        retrieved_ids = extract_doc_ids_from_results(candidates)

        source_docs = pipeline._rerank(sample.query, candidates)

        if retrieval_only:
            answer = ""
            faith = 0.0
            acc = 0.0
        else:
            from src.utils.memory import clear_memory_cache
            clear_memory_cache(device)
            system_prompt, user_prompt = build_rag_prompt(sample.query, source_docs)
            answer = pipeline.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
            )
            faith = compute_faithfulness(
                answer=answer,
                context_docs=source_docs,
                llm_client=pipeline.llm_client,
            )
            acc = compute_answer_accuracy(answer, sample.reference_answer)

        if mode == "adaptive_hyde" and route == "hyde":
            hyde_trigger_count += 1
        if mode == "hybrid" and route == "hyde":
            hyde_trigger_count += 1

        latency_ms = (time.perf_counter() - t0) * 1000

        all_retrieved_ids.append(retrieved_ids)
        all_relevant_ids.append(sample.relevant_doc_ids)
        faithfulness_scores.append(faith)
        answer_accuracy_scores.append(acc)
        latencies.append(latency_ms)

        detail = {
            "query": sample.query,
            "doc_id": sample.doc_id or (sample.relevant_doc_ids[0] if sample.relevant_doc_ids else ""),
            "reference_answer": sample.reference_answer,
            "answer_preview": answer[:100] if answer else "",
            "answer_accuracy": acc,
            "faithfulness": faith,
            "latency_ms": round(latency_ms, 1),
            "retrieved_ids": retrieved_ids[:5],
            "hit_at_5": bool(set(retrieved_ids[:5]) & set(sample.relevant_doc_ids)),
        }
        if route is not None:
            detail["route"] = route
            if mode == "adaptive_hyde" and pipeline.adaptive_hyde_retriever:
                detail["route_reason"] = pipeline.adaptive_hyde_retriever.last_route_reason
        details.append(detail)

        if (i + 1) % 10 == 0:
            logger.info(f"评估进度: {i+1}/{len(samples)}")

        if gc_interval > 0 and (i + 1) % gc_interval == 0:
            from src.utils.memory import clear_memory_cache
            clear_memory_cache(device)

    mrr = compute_mrr_at_k(all_retrieved_ids, all_relevant_ids, k=5)
    recall = compute_recall_at_k(all_retrieved_ids, all_relevant_ids, k=5)
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_acc = sum(answer_accuracy_scores) / len(answer_accuracy_scores) if answer_accuracy_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    hyde_trigger_rate = None
    if mode in ("adaptive_hyde", "hybrid") and samples:
        hyde_trigger_rate = round(hyde_trigger_count / len(samples), 4)

    report = EvalReport(
        mode=mode,
        mrr_at_5=mrr,
        recall_at_5=recall,
        avg_faithfulness=round(avg_faith, 4),
        avg_answer_accuracy=round(avg_acc, 4),
        avg_latency_ms=round(avg_latency, 1),
        num_samples=len(samples),
        hyde_trigger_rate=hyde_trigger_rate,
        details=details,
    )

    trigger_info = f" | HyDE触发率={hyde_trigger_rate:.2%}" if hyde_trigger_rate is not None else ""
    logger.info(
        f"[{mode.upper()}] MRR@5={mrr:.4f} | Recall@5={recall:.4f} | "
        f"AnswerAcc={avg_acc:.4f} | Faithfulness={avg_faith:.4f} | "
        f"AvgLatency={avg_latency:.1f}ms{trigger_info}"
    )
    return report


def _report_to_dict(report: EvalReport) -> dict:
    """序列化报告，details 仅保留前 DETAIL_LIMIT 条（指标仍基于全量样本）。"""
    data = asdict(report)
    data["details"] = data["details"][:DETAIL_LIMIT]
    return data


def _save_naive_hits(naive_report: EvalReport, hits_path: str) -> None:
    """Naive 评测完成后落盘全量 hit 明细，供 Hard 子集导出复用。"""
    Path(hits_path).parent.mkdir(parents=True, exist_ok=True)
    with open(hits_path, "w", encoding="utf-8") as f:
        for d in naive_report.details:
            f.write(json.dumps({
                "query": d["query"],
                "doc_id": d["doc_id"],
                "hit_at_5": d["hit_at_5"],
                "retrieved_ids": d.get("retrieved_ids", []),
            }, ensure_ascii=False) + "\n")
    logger.info(f"Naive hit 明细已保存: {hits_path}（{len(naive_report.details)} 条）")


def _metrics_on_indices(report: EvalReport, indices: List[int]) -> dict:
    """按样本下标子集重算 MRR@5 / Recall@5。"""
    if not indices:
        return {"count": 0, "mrr_at_5": 0.0, "recall_at_5": 0.0}

    retrieved = [report.details[i]["retrieved_ids"] for i in indices]
    relevant = [
        [report.details[i]["doc_id"]] if report.details[i]["doc_id"] else []
        for i in indices
    ]
    return {
        "count": len(indices),
        "mrr_at_5": compute_mrr_at_k(retrieved, relevant, k=5),
        "recall_at_5": compute_recall_at_k(retrieved, relevant, k=5),
    }


def _build_bucket_report(reports: dict) -> Optional[dict]:
    """按 Naive hit@5 划分 Easy / Hard 桶，对比各模式 MRR@5。"""
    if "naive" not in reports:
        return None

    naive_details = reports["naive"].details
    hard_idx = [i for i, d in enumerate(naive_details) if not d["hit_at_5"]]
    easy_idx = [i for i, d in enumerate(naive_details) if d["hit_at_5"]]
    total = len(naive_details) or 1

    bucket = {
        "hard": {"count": len(hard_idx), "ratio": round(len(hard_idx) / total, 4)},
        "easy": {"count": len(easy_idx), "ratio": round(len(easy_idx) / total, 4)},
        "by_mode": {},
    }

    for mode, report in reports.items():
        bucket["by_mode"][mode] = {
            "all": {
                "mrr_at_5": report.mrr_at_5,
                "recall_at_5": report.recall_at_5,
            },
            "hard": _metrics_on_indices(report, hard_idx),
            "easy": _metrics_on_indices(report, easy_idx),
        }
    return bucket


def _build_summary(reports: dict, eval_modes: tuple) -> Optional[dict]:
    """全部模式完成后计算 summary；否则返回 None。"""
    if not all(mode in reports for mode in eval_modes):
        return None

    if not all(m in reports for m in ("naive", "hyde", "adaptive_hyde", "hybrid")):
        return None

    naive_report = reports["naive"]
    hyde_report = reports["hyde"]
    adaptive_report = reports["adaptive_hyde"]
    hybrid_report = reports["hybrid"]

    def _gain_pct(baseline: float, improved: float) -> Optional[float]:
        """baseline 为 0 时百分比无意义，返回 null。"""
        if baseline < 1e-9:
            return None
        return round((improved - baseline) / baseline * 100, 1)

    return {
        "hyde_vs_naive_mrr_gain_pct": _gain_pct(naive_report.mrr_at_5, hyde_report.mrr_at_5),
        "hyde_vs_naive_mrr_abs": round(hyde_report.mrr_at_5 - naive_report.mrr_at_5, 4),
        "adaptive_vs_naive_mrr_gain_pct": _gain_pct(
            naive_report.mrr_at_5, adaptive_report.mrr_at_5
        ),
        "adaptive_vs_naive_mrr_abs": round(adaptive_report.mrr_at_5 - naive_report.mrr_at_5, 4),
        "hybrid_vs_naive_mrr_gain_pct": _gain_pct(
            naive_report.mrr_at_5, hybrid_report.mrr_at_5
        ),
        "hybrid_vs_naive_mrr_abs": round(hybrid_report.mrr_at_5 - naive_report.mrr_at_5, 4),
        "adaptive_vs_hyde_mrr_gain_pct": _gain_pct(
            hyde_report.mrr_at_5, adaptive_report.mrr_at_5
        ),
        "adaptive_hyde_trigger_rate": adaptive_report.hyde_trigger_rate,
        "hybrid_hyde_trigger_rate": hybrid_report.hyde_trigger_rate,
        "adaptive_latency_saved_vs_hyde_ms": round(
            hyde_report.avg_latency_ms - adaptive_report.avg_latency_ms, 1
        ),
    }


def _save_report(
    reports: dict,
    output_path: str,
    completed_modes: List[str],
    eval_modes: tuple = DEFAULT_EVAL_MODES,
) -> dict:
    """将当前已完成模式的评测结果写入文件（增量 checkpoint）。"""
    comparison = {
        mode: _report_to_dict(reports[mode]) if mode in reports else None
        for mode in eval_modes
    }
    summary = _build_summary(reports, eval_modes)
    comparison["summary"] = summary
    if "naive" in reports and len(completed_modes) == len(eval_modes):
        comparison["buckets"] = _build_bucket_report(reports)
    comparison["_meta"] = {
        "completed_modes": completed_modes,
        "pending_modes": [m for m in eval_modes if m not in completed_modes],
        "all_complete": len(completed_modes) == len(eval_modes),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info(
        f"评测进度已保存: {output_path} "
        f"({len(completed_modes)}/{len(eval_modes)} 模式: {', '.join(completed_modes)})"
    )
    return comparison


def compare_and_report(
    pipeline: RAGPipeline,
    eval_dataset_path: str,
    output_path: str = "data/eval_report.json",
    retrieval_only: bool = False,
    limit: Optional[int] = None,
    gc_interval: int = 20,
    device: str = "mps",
    eval_modes: tuple = DEFAULT_EVAL_MODES,
) -> dict:
    """
    对比 Naive / HyDE / Adaptive HyDE，生成完整对比报告。

    Args:
        pipeline: RAG Pipeline 实例
        eval_dataset_path: 评估数据集路径（JSONL）
        output_path: 报告输出路径（JSON）
        retrieval_only: 仅评估检索指标
        limit: 最多评估样本数

    Returns:
        包含两种策略评估结果的字典
    """
    samples = load_eval_dataset(eval_dataset_path, limit=limit)
    hits_path = str(Path(output_path).with_name(
        Path(eval_dataset_path).stem + "_naive_hits.jsonl"
    ))

    reports: dict = {}
    completed_modes: List[str] = []
    comparison: dict = {}

    for mode in eval_modes:
        reports[mode] = run_eval(
            pipeline, samples, mode=mode,
            retrieval_only=retrieval_only, gc_interval=gc_interval, device=device,
        )
        if mode == "naive":
            _save_naive_hits(reports["naive"], hits_path)
        completed_modes.append(mode)
        comparison = _save_report(reports, output_path, completed_modes, eval_modes)

    summary = comparison.get("summary")
    if summary:
        logger.info(
            f"对比报告已完成: {output_path} | "
            f"HyDE vs Naive MRR@5 +{summary['hyde_vs_naive_mrr_gain_pct']}% | "
            f"Adaptive vs Naive +{summary['adaptive_vs_naive_mrr_gain_pct']}% | "
            f"Adaptive HyDE 触发率 {summary['adaptive_hyde_trigger_rate']}"
        )

    return comparison


def _build_pipeline_from_config(cfg: dict) -> RAGPipeline:
    return build_pipeline(cfg)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Naive / HyDE / Adaptive HyDE 对比评测")
    parser.add_argument("--eval_data", default="data/eval_dataset.jsonl")
    parser.add_argument("--output", default="data/eval_report.json")
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--limit", type=int, default=None, help="最多评估样本数")
    parser.add_argument(
        "--retrieval_only",
        action="store_true",
        help="仅评估检索指标（MRR/Recall），跳过 LLM 生成",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="逗号分隔评测模式，默认全部: naive,hyde,adaptive_hyde,hybrid",
    )
    args = parser.parse_args()

    eval_modes = DEFAULT_EVAL_MODES
    if args.modes:
        eval_modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipeline = _build_pipeline_from_config(cfg)
    memory_cfg = cfg.get("memory", {})
    embedding_cfg = cfg.get("embedding", {})
    compare_and_report(
        pipeline,
        args.eval_data,
        args.output,
        retrieval_only=args.retrieval_only,
        limit=args.limit,
        gc_interval=memory_cfg.get("eval_gc_interval", 20),
        device=embedding_cfg.get("device", "mps"),
        eval_modes=eval_modes,
    )
