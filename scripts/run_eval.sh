#!/bin/bash
# Naive RAG vs HyDE 对比评测（低内存模式）
#
# 用法:
#   bash scripts/run_eval.sh --retrieval_only --limit 50   # 推荐先试跑
#   bash scripts/run_eval.sh --limit 100
#   bash scripts/run_eval.sh --modes naive,hyde,hybrid       # 跳过 adaptive
#   EVAL_DATA=data/eval_dataset_200_hard.jsonl bash scripts/run_eval.sh --retrieval_only

set -e
source "$(dirname "$0")/env.sh"

CONFIG="configs/config.local.yaml"
EVAL_DATA="${EVAL_DATA:-data/eval_dataset.jsonl}"
OUTPUT="${OUTPUT:-data/eval_report.json}"
EXTRA_ARGS="$@"

echo "======================================"
echo "  Naive / HyDE / Adaptive HyDE 三路对比评测"
echo "  Python: $PROJECT_PYTHON"
echo "  配置: $CONFIG"
echo "======================================"

if [[ ! -f "$EVAL_DATA" ]]; then
  echo "⚠️  评测集不存在，先构建..."
  bash scripts/prepare_eval.sh
fi

if [[ ! -d "data/faiss_index" ]]; then
  echo "⚠️  FAISS 索引不存在，开始构建（流式低内存模式）..."
  bash scripts/build_index.sh data/eval_docs/docs
fi

# 确保 Ollama 可用并预热 LLM（避免首次 chat 502）
OLLAMA_URL="http://127.0.0.1:11434"
if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null; then
  echo "⚠️  Ollama 未运行，尝试启动..."
  if command -v ollama &>/dev/null; then
    OLLAMA_KEEP_ALIVE=-1 ollama serve >/tmp/ollama-serve.log 2>&1 &
    sleep 5
  fi
fi
if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null; then
  LLM_MODEL=$("$PROJECT_PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['llm']['model_name'])")
  echo "🔥 预热 Ollama 模型: $LLM_MODEL"
  curl -sf "${OLLAMA_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${LLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false}" \
    >/dev/null || echo "⚠️  预热失败，继续尝试评测..."
else
  echo "❌ Ollama 不可用，请先运行: ollama serve && ollama pull qwen2.5:1.5b"
  exit 1
fi

export TOKENIZERS_PARALLELISM=false

"$PROJECT_PYTHON" -m src.evaluation.compare \
  --config "$CONFIG" \
  --eval_data "$EVAL_DATA" \
  --output "$OUTPUT" \
  $EXTRA_ARGS

echo "✅ 评测完成，报告: $OUTPUT"
