#!/bin/bash
# 一键构建 FAISS 向量索引（流式分批，适合 Mac 24GB）
# 用法: bash scripts/build_index.sh [docs_dir]

set -e
source "$(dirname "$0")/env.sh"

DOCS_DIR="${1:-data/eval_docs/docs}"
CONFIG="configs/config.local.yaml"

echo "======================================"
echo "  RAG 知识库 - 向量索引构建（低内存模式）"
echo "  Python: $PROJECT_PYTHON"
echo "  文档目录: $DOCS_DIR"
echo "======================================"

"$PROJECT_PYTHON" - <<EOF
import yaml
from src.ingestion.indexer import build_index_from_directory

with open("$CONFIG", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

ingestion = cfg["ingestion"]
embedding = cfg["embedding"]
memory = cfg.get("memory", {})

# 建索引可用独立设备（cpu 更稳、更省 MPS 内存；推理仍用 embedding.device）
build_device = embedding.get("index_build_device") or embedding["device"]

print(f"建索引设备: {build_device}（推理设备: {embedding['device']}）")

build_index_from_directory(
    docs_dir="$DOCS_DIR",
    index_path=ingestion["index_path"],
    supported_formats=ingestion["supported_formats"],
    chunk_size=ingestion["chunk_size"],
    chunk_overlap=ingestion["chunk_overlap"],
    model_name=embedding["model_name"],
    device=build_device,
    embed_batch_size=embedding["batch_size"],
    doc_batch_size=memory.get("doc_batch_size", 32),
)

print("✅ 索引构建完成！")
EOF
