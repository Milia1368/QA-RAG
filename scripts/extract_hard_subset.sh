#!/bin/bash
# 扫描 Naive 粗排结果，导出 Hard / Easy 评测子集
#
# 用法:
#   bash scripts/extract_hard_subset.sh
#   bash scripts/extract_hard_subset.sh data/eval_dataset_200.jsonl

set -e
source "$(dirname "$0")/env.sh"

DATASET="${1:-data/eval_dataset_200.jsonl}"
STEM=$(basename "$DATASET" .jsonl)

echo "======================================"
echo "  导出 Hard / Easy 子集"
echo "  数据集: $DATASET"
echo "======================================"

"$PROJECT_PYTHON" -m src.evaluation.extract_hard_subset \
  --dataset "$DATASET" \
  --output "data/${STEM}_hard.jsonl" \
  --output_easy "data/${STEM}_easy.jsonl" \
  --hits_cache "data/${STEM}_naive_hits.jsonl"

echo "✅ Hard 子集: data/${STEM}_hard.jsonl"
echo "   评测: EVAL_DATA=data/${STEM}_hard.jsonl bash scripts/run_eval.sh --retrieval_only"
