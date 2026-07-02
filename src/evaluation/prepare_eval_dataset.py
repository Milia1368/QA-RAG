"""
从 QuestAnswer1Doc 标注 JSON 构建 eval_dataset.jsonl。

输入格式（QuestAnswer1Doc_quest_gt_save.json）：
    {
        "<doc_id>": {
            "question": ["问题1", ...],
            "answers": ["答案1", ...],
            "key_info": [...]
        },
        ...
    }

输出格式（JSONL，每行一条）：
    {
        "query": "...",
        "relevant_doc_ids": ["<doc_id>"],
        "reference_answer": "...",
        "doc_id": "<doc_id>"
    }
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def build_eval_dataset(
    qa_json_path: str,
    docs_dir: str,
    output_path: str,
    skip_uninferable: bool = False,
    limit: Optional[int] = None,
) -> int:
    """
    将 QuestAnswer1Doc 标注转为评测 JSONL。

    Args:
        qa_json_path: 标注 JSON 路径
        docs_dir: 文档目录（用于校验 doc_id 对应文件是否存在）
        output_path: 输出 JSONL 路径
        skip_uninferable: 是否跳过参考答案为「无法推断」的样本
        limit: 最多输出条数（None 表示全部）

    Returns:
        写入的样本数
    """
    qa_path = Path(qa_json_path)
    docs_path = Path(docs_dir)
    out_path = Path(output_path)

    if not qa_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {qa_json_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    available_doc_ids = {p.stem for p in docs_path.glob("*.txt")} if docs_path.exists() else set()

    samples: List[dict] = []
    skipped_missing_doc = 0
    skipped_uninferable = 0
    skipped_mismatch = 0

    for doc_id, entry in qa_data.items():
        if available_doc_ids and doc_id not in available_doc_ids:
            skipped_missing_doc += len(entry.get("question", []))
            continue

        questions = entry.get("question", [])
        answers = entry.get("answers", [])

        if len(questions) != len(answers):
            logger.warning(
                f"doc_id={doc_id} 问答数量不一致: "
                f"{len(questions)} questions vs {len(answers)} answers"
            )
            skipped_mismatch += abs(len(questions) - len(answers))
            pair_count = min(len(questions), len(answers))
        else:
            pair_count = len(questions)

        for i in range(pair_count):
            answer = answers[i].strip()
            if skip_uninferable and answer == "无法推断":
                skipped_uninferable += 1
                continue

            samples.append({
                "query": questions[i].strip(),
                "relevant_doc_ids": [doc_id],
                "reference_answer": answer,
                "doc_id": doc_id,
            })

            if limit is not None and len(samples) >= limit:
                break

        if limit is not None and len(samples) >= limit:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        f"评测集已写入: {output_path}，共 {len(samples)} 条 | "
        f"跳过无文档 {skipped_missing_doc} 条 | "
        f"跳过无法推断 {skipped_uninferable} 条 | "
        f"跳过问答不匹配 {skipped_mismatch} 条"
    )
    return len(samples)


def sample_eval_dataset(
    input_path: str,
    output_path: str,
    size: int,
    seed: int = 42,
) -> int:
    """
    从已有评测 JSONL 中随机抽取若干条，写入新文件（可复现）。

    Args:
        input_path: 源 JSONL 路径
        output_path: 输出 JSONL 路径
        size: 抽样条数
        seed: 随机种子

    Returns:
        实际写入条数
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"源评测集不存在: {input_path}")

    samples: List[dict] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"源评测集为空: {input_path}")

    if size > len(samples):
        logger.warning(
            f"请求抽样 {size} 条，但源集仅 {len(samples)} 条，将全部保留"
        )
        selected = samples
    else:
        selected = random.Random(seed).sample(samples, size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        f"随机抽样完成: {output_path}，共 {len(selected)}/{len(samples)} 条 "
        f"(seed={seed})"
    )
    return len(selected)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="构建 RAG 评测数据集 JSONL")
    parser.add_argument(
        "--qa_json",
        default="data/eval_docs/eval_ans/QuestAnswer1Doc_quest_gt_save.json",
        help="QuestAnswer1Doc 标注 JSON 路径",
    )
    parser.add_argument(
        "--docs_dir",
        default="data/eval_docs/docs",
        help="文档目录（用于校验 doc_id）",
    )
    parser.add_argument(
        "--output",
        default="data/eval_dataset.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--include_uninferable",
        action="store_true",
        help="保留参考答案为「无法推断」的样本",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多输出样本数（用于快速试跑）",
    )
    parser.add_argument(
        "--sample_from",
        default=None,
        help="从已有 JSONL 随机抽样（指定源文件路径，与 --qa_json 互斥）",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="随机抽样条数（配合 --sample_from 使用，默认 200）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子（默认 42，保证可复现）",
    )
    args = parser.parse_args()

    if args.sample_from:
        sample_eval_dataset(
            input_path=args.sample_from,
            output_path=args.output,
            size=args.sample_size,
            seed=args.seed,
        )
        return

    build_eval_dataset(
        qa_json_path=args.qa_json,
        docs_dir=args.docs_dir,
        output_path=args.output,
        skip_uninferable=not args.include_uninferable,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
