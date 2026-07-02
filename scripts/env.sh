#!/bin/bash
# 项目公共环境：conda + Python 路径
#
# 用法（在其他脚本开头）:
#   source "$(dirname "$0")/env.sh"
#
# MPS 内存水位线（可选，默认不设置）:
#   PyTorch 默认 LOW=1.4（统一内存 Mac），若只把 HIGH 设成 0.65 会触发
#   "invalid low watermark ratio 1.4"。必须同时设置且 LOW <= HIGH，例如:
#     export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
#     export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 仅设置 HIGH 而未设置 LOW 时，Mac 默认 LOW=1.4 会大于 HIGH，导致模型加载崩溃
if [[ -n "${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-}" && -z "${PYTORCH_MPS_LOW_WATERMARK_RATIO:-}" ]]; then
    echo "⚠️  已忽略 PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO}（需同时设置 LOW 且 LOW<=HIGH，否则报错 invalid low watermark ratio 1.4）"
    unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
fi

# 激活 conda 环境（优先 langchain-rag，兼容用户口述的 lang-chain）
TARGET_ENV="${CONDA_ENV:-langchain-rag}"
FALLBACK_ENV="lang-chain"

if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        if [[ "${CONDA_DEFAULT_ENV:-}" != "$TARGET_ENV" ]]; then
            if conda activate "$TARGET_ENV" 2>/dev/null; then
                :
            elif [[ "$TARGET_ENV" != "$FALLBACK_ENV" ]] && conda activate "$FALLBACK_ENV" 2>/dev/null; then
                TARGET_ENV="$FALLBACK_ENV"
            else
                echo "⚠️  未找到 conda 环境 ${TARGET_ENV}，将使用当前 Python"
            fi
        fi
    fi
fi

PYTHON="${CONDA_PREFIX:+$CONDA_PREFIX/bin/}python"
if ! command -v "$PYTHON" &>/dev/null; then
    PYTHON="python"
fi

export PROJECT_PYTHON="$PYTHON"
export PROJECT_CONDA_ENV="${CONDA_DEFAULT_ENV:-unknown}"
