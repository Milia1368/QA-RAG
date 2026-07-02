#!/bin/bash
# 从 QuestAnswer1Doc 标注构建评测 JSONL
# 用法: bash scripts/prepare_eval.sh [--limit N]

set -e
source "$(dirname "$0")/env.sh"

LIMIT=""
if [[ "$1" == "--limit" && -n "$2" ]]; then
  LIMIT="--limit $2"
fi

echo "======================================"
echo "  构建评测数据集 eval_dataset.jsonl"
echo "  Python: $PROJECT_PYTHON"
echo "======================================"

"$PROJECT_PYTHON" -m src.evaluation.prepare_eval_dataset \
  --qa_json data/eval_docs/eval_ans/QuestAnswer1Doc_quest_gt_save.json \
  --docs_dir data/eval_docs/docs \
  --output data/eval_dataset.jsonl --include_uninferable \
  $LIMIT

echo "✅ 评测集构建完成: data/eval_dataset.jsonl"
