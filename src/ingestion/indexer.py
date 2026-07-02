"""
FAISS 向量索引构建与加载模块
使用 BGE-large-zh 作为 Embedding 模型，支持增量追加与持久化。
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from src.utils.memory import batched, clear_memory_cache


def _build_embeddings(model_name: str, device: str = "mps") -> HuggingFaceEmbeddings:
    """初始化 BGE Embedding 模型。"""
    encode_kwargs = {"normalize_embeddings": True}  # BGE 建议归一化
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
    )
    logger.info(f"Embedding 模型加载完成: {model_name} on {device}")
    return embeddings


def _add_chunks_to_index(
    vectorstore: Optional[FAISS],
    embeddings: HuggingFaceEmbeddings,
    chunks: List[Document],
    embed_batch_size: int,
    device: str,
) -> FAISS:
    """分批向量化 chunks 并追加到索引，每批后释放 MPS 缓存。"""
    for batch in batched(chunks, embed_batch_size):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        clear_memory_cache(device)
    if vectorstore is None:
        raise ValueError("没有可索引的 chunk，请检查文档目录是否为空")
    return vectorstore


def build_index(
    chunks: List[Document],
    index_path: str,
    model_name: str = "BAAI/bge-large-zh-v1.5",
    device: str = "mps",
    batch_size: int = 8,
) -> FAISS:
    """
    从 chunk 列表构建 FAISS 向量索引并持久化到磁盘。

    Args:
        chunks: 切分后的 Document 列表
        index_path: 索引保存目录
        model_name: Embedding 模型名称
        device: 推理设备
        batch_size: 向量化批大小（Mac 24GB 建议 4~8）

    Returns:
        构建好的 FAISS VectorStore 对象
    """
    logger.info(f"开始构建 FAISS 索引，共 {len(chunks)} 个 chunks，embed_batch={batch_size}...")
    embeddings = _build_embeddings(model_name, device)
    vectorstore = _add_chunks_to_index(None, embeddings, chunks, batch_size, device)

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info(f"FAISS 索引已保存至: {index_path}")
    return vectorstore


def build_index_from_directory(
    docs_dir: str,
    index_path: str,
    supported_formats: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    model_name: str = "BAAI/bge-large-zh-v1.5",
    device: str = "mps",
    embed_batch_size: int = 8,
    doc_batch_size: int = 32,
) -> FAISS:
    """
    流式分批建索引：每次只加载 doc_batch_size 个文件，避免 2000+ 文档同时驻留内存。

    适合 Mac 24GB 等内存受限环境。
    """
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_document

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    files = sorted(
        str(f)
        for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_formats
    )
    if not files:
        raise FileNotFoundError(f"目录下没有支持的文档: {docs_dir}")

    logger.info(
        f"流式建索引: {len(files)} 个文件, "
        f"doc_batch={doc_batch_size}, embed_batch={embed_batch_size}"
    )

    embeddings = _build_embeddings(model_name, device)
    vectorstore: Optional[FAISS] = None
    total_chunks = 0

    for file_idx, file_batch in enumerate(batched(files, doc_batch_size), start=1):
        batch_docs: List[Document] = []
        for file_path in file_batch:
            try:
                batch_docs.extend(load_document(file_path))
            except Exception as e:
                logger.warning(f"加载失败: {file_path}，原因: {e}")

        if not batch_docs:
            continue

        batch_chunks = chunk_documents(
            batch_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total_chunks += len(batch_chunks)
        vectorstore = _add_chunks_to_index(
            vectorstore, embeddings, batch_chunks, embed_batch_size, device
        )

        del batch_docs, batch_chunks
        clear_memory_cache(device)
        logger.info(
            f"文档批次 {file_idx}/{(len(files) + doc_batch_size - 1) // doc_batch_size} 完成，"
            f"累计 chunks: {total_chunks}"
        )

    if vectorstore is None:
        raise ValueError("索引构建失败：未成功加载任何文档")

    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info(f"FAISS 索引已保存至: {index_path}，共 {total_chunks} chunks")
    return vectorstore


def load_index(
    index_path: str,
    model_name: str = "BAAI/bge-large-zh-v1.5",
    device: str = "mps",
) -> FAISS:
    """
    从磁盘加载已有 FAISS 索引。

    Args:
        index_path: 索引目录
        model_name: 与构建时一致的 Embedding 模型
        device: 推理设备

    Returns:
        加载好的 FAISS VectorStore 对象
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引目录不存在: {index_path}，请先运行 build_index")

    embeddings = _build_embeddings(model_name, device)
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"FAISS 索引加载成功: {index_path}")
    return vectorstore
