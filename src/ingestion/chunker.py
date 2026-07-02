"""
文档 Chunk 切分模块
策略：512 token + 64 token overlap，基于 tiktoken 计算长度。
"""

from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


# 使用 cl100k_base（GPT-4/Qwen 兼容编码）计算 token 数
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """计算文本 token 数量。"""
    return len(_TOKENIZER.encode(text))


def build_chunker(chunk_size: int = 512, chunk_overlap: int = 64) -> RecursiveCharacterTextSplitter:
    """
    构造基于 token 计数的递归字符切分器。

    分隔符优先级（中文场景优化）：
      段落换行 → 单换行 → 句号/问号/感叹号 → 逗号 → 空格 → 字符级

    Args:
        chunk_size: 每个 chunk 的最大 token 数（默认 512）
        chunk_overlap: 相邻 chunk 的重叠 token 数（默认 64）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
        separators=[
            "\n\n",   # 段落
            "\n",     # 换行
            "。",     # 中文句号
            "！",
            "？",
            "；",
            "，",
            " ",
            "",       # 字符级回退
        ],
        is_separator_regex=False,
    )
    return splitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    对文档列表执行 chunk 切分，保留原始 metadata。

    Args:
        docs: 原始 Document 列表
        chunk_size: chunk token 上限
        chunk_overlap: overlap token 数

    Returns:
        切分后的 Document 列表，每个 chunk 附加 chunk_index 元数据
    """
    splitter = build_chunker(chunk_size, chunk_overlap)
    chunks: List[Document] = []

    for doc in docs:
        sub_docs = splitter.split_documents([doc])
        for i, sub in enumerate(sub_docs):
            sub.metadata["chunk_index"] = i
            sub.metadata["chunk_total"] = len(sub_docs)
            sub.metadata["token_count"] = _token_len(sub.page_content)
        chunks.extend(sub_docs)

    logger.info(
        f"切分完成：{len(docs)} 文档 → {len(chunks)} chunks，"
        f"策略: chunk_size={chunk_size}, overlap={chunk_overlap}"
    )
    return chunks
