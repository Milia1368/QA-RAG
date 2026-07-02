"""
从评测集 + Naive 检索结果导出 Easy / Hard 子集。

Hard = Naive 粗排 top-5 未命中标注 doc_id
Easy = Naive 粗排 top-5 已命中
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.evaluation.compare import EvalSample, load_eval_dataset
from src.evaluation.metrics import extract_doc_ids_from_results
from src.generation.factory import build_pipeline


def scan_naive_hits(
    pipeline,
    samples: List[EvalSample],
) -> List[dict]:
    """对每条样本跑 Naive 粗排，返回 hit 明细（全量，不截断）。"""
    hits: List[dict] = []
    for i, sample in enumerate(samples):
        candidates = pipeline.naive_retriever.retrieve_docs(
            sample.query, top_k=pipeline.retrieval_top_k
        )
        retrieved_ids = extract_doc_ids_from_results(candidates)
        hit_at_5 = bool(set(retrieved_ids[:5]) & set(sample.relevant_doc_ids))
        hits.append({
            "query": sample.query,
            "doc_id": sample.doc_id or (sample.relevant_doc_ids[0] if sample.relevant_doc_ids else ""),
            "hit_at_5": hit_at_5,
            "retrieved_ids": retrieved_ids[:5],
        })
        if (i + 1) % 50 == 0:
            logger.info(f"Naive 扫描进度: {i + 1}/{len(samples)}")
    return hits


def _load_dataset_by_query(path: str) -> Dict[str, dict]:
    by_query: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_query[row["query"]] = row
    return by_query


def export_subsets(
    dataset_path: str,
    hits: List[dict],
    output_hard: str,
    output_easy: Optional[str] = None,
    hits_cache: Optional[str] = None,
) -> dict:
    """按 hit_at_5 将原 JSONL 拆为 Hard / Easy 子集。"""
    by_query = _load_dataset_by_query(dataset_path)
    hard_rows: List[dict] = []
    easy_rows: List[dict] = []

    for item in hits:
        row = by_query.get(item["query"])
        if row is None:
            logger.warning(f"未在数据集中找到 query: {item['query'][:40]}...")
            continue
        if item["hit_at_5"]:
            easy_rows.append(row)
        else:
            hard_rows.append(row)

    Path(output_hard).parent.mkdir(parents=True, exist_ok=True)
    with open(output_hard, "w", encoding="utf-8") as f:
        for row in hard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    easy_path = output_easy
    if easy_path:
        with open(easy_path, "w", encoding="utf-8") as f:
            for row in easy_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if hits_cache:
        Path(hits_cache).parent.mkdir(parents=True, exist_ok=True)
        with open(hits_cache, "w", encoding="utf-8") as f:
            for item in hits:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "total": len(hits),
        "hard": len(hard_rows),
        "easy": len(easy_rows),
        "hard_ratio": round(len(hard_rows) / len(hits), 4) if hits else 0.0,
        "output_hard": output_hard,
        "output_easy": easy_path,
        "hits_cache": hits_cache,
    }
    logger.info(
        f"子集导出完成: Hard={stats['hard']} Easy={stats['easy']} "
        f"(Hard 占比 {stats['hard_ratio']:.1%}) → {output_hard}"
    )
    return stats


def load_hits_from_cache(path: str) -> List[dict]:
    hits: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hits.append(json.loads(line))
    return hits


def main():
    parser = argparse.ArgumentParser(description="导出 Naive Hard/Easy 评测子集")
    parser.add_argument("--dataset", default="data/eval_dataset_200.jsonl")
    parser.add_argument("--output", default="data/eval_dataset_hard.jsonl")
    parser.add_argument("--output_easy", default="data/eval_dataset_easy.jsonl")
    parser.add_argument("--hits_cache", default="data/eval_naive_hits.jsonl")
    parser.add_argument(
        "--hits_from",
        default=None,
        help="已有 hits 缓存路径，指定则跳过 Naive 扫描",
    )
    parser.add_argument("--config", default="configs/config.local.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    samples = load_eval_dataset(args.dataset, limit=args.limit)

    if args.hits_from:
        hits = load_hits_from_cache(args.hits_from)
        logger.info(f"从缓存加载 hits: {args.hits_from}，共 {len(hits)} 条")
    else:
        pipeline = build_pipeline(cfg)
        hits = scan_naive_hits(pipeline, samples)

    stats = export_subsets(
        args.dataset,
        hits,
        output_hard=args.output,
        output_easy=args.output_easy,
        hits_cache=args.hits_cache if not args.hits_from else None,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
