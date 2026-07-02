#!/bin/bash
# 从全量评测集随机抽取子集（默认 200 条）
#
# 用法:
#   bash scripts/sample_eval.sh                  # → data/eval_dataset_200.jsonl
#   bash scripts/sample_eval.sh 100              # → data/eval_dataset_100.jsonl
#   bash scripts/sample_eval.sh 200 42           # 指定条数与随机种子

set -e
source "$(dirname "$0")/env.sh"

SIZE="${1:-200}"
SEED="${2:-42}"
SOURCE="data/eval_dataset.jsonl"
OUTPUT="data/eval_dataset_${SIZE}.jsonl"

echo "======================================"
echo "  随机抽取评测子集"
echo "  源: $SOURCE"
echo "  输出: $OUTPUT (${SIZE} 条, seed=${SEED})"
echo "======================================"

if [[ ! -f "$SOURCE" ]]; then
  echo "⚠️  全量评测集不存在，先构建..."
  bash scripts/prepare_eval.sh
fi

"$PROJECT_PYTHON" -m src.evaluation.prepare_eval_dataset \
  --sample_from "$SOURCE" \
  --output "$OUTPUT" \
  --sample_size "$SIZE" \
  --seed "$SEED"

echo "✅ 子集已生成: $OUTPUT"
echo "   运行评测: EVAL_DATA=$OUTPUT bash scripts/run_eval.sh"
