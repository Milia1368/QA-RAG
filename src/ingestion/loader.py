"""
文档加载模块
支持 PDF、DOCX、TXT、Markdown 多格式，统一转为 LangChain Document 对象。
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from loguru import logger


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_document(file_path: str) -> List[Document]:
    suffix = Path(file_path).suffix.lower()
    if suffix not in LOADER_MAP:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: {list(LOADER_MAP.keys())}")

    loader_cls = LOADER_MAP[suffix]
    # 关键区分：txt 文件强制 utf-8 编码加载
    if suffix == ".txt":
        loader = loader_cls(file_path, encoding="utf-8")
    else:
        loader = loader_cls(file_path)
        
    docs = loader.load()

    doc_id = Path(file_path).stem
    for doc in docs:
        doc.metadata.setdefault("source", file_path)
        doc.metadata.setdefault("title", doc_id)
        doc.metadata.setdefault("doc_id", doc_id)

    logger.info(f"加载文档: {file_path}，共 {len(docs)} 段")
    return docs



def load_directory(docs_dir: str, supported_formats: List[str] = None) -> List[Document]:
    """
    递归加载目录下所有支持格式的文档。

    Args:
        docs_dir: 文档根目录
        supported_formats: 支持的文件后缀列表，默认全部支持格式

    Returns:
        合并后的 Document 列表
    """
    if supported_formats is None:
        supported_formats = list(LOADER_MAP.keys())

    all_docs: List[Document] = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    files = [
        str(f)
        for f in docs_path.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_formats
    ]

    logger.info(f"发现 {len(files)} 个文档文件，开始加载...")

    for file_path in files:
        try:
            docs = load_document(file_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"加载失败: {file_path}，原因: {e}")

    logger.info(f"共加载 {len(all_docs)} 个文档段落")
    return all_docs
