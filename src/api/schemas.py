"""
FastAPI 请求 / 响应 Pydantic 模型定义
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求体"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    mode: Literal["naive", "hyde", "adaptive_hyde", "hybrid"] = Field(
        default="hybrid",
        description="检索策略：naive / hyde / adaptive_hyde / hybrid(两阶段并集)",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="检索文档数量")


class SourceDoc(BaseModel):
    """来源文档摘要"""
    source: str = Field(description="来源文件路径")
    chunk_index: int = Field(description="在原文档中的段落序号")
    snippet: str = Field(description="文档片段预览（前200字）")


class ChatResponse(BaseModel):
    """问答响应体"""
    answer: str = Field(description="生成的答案")
    sources: List[SourceDoc] = Field(description="答案溯源文档列表")
    retrieval_mode: str = Field(description="本次使用的检索策略")
    query: str = Field(description="原始问题（原样返回，便于对账）")
    cached: bool = Field(default=False, description="是否命中 Redis 缓存")


class IndexBuildRequest(BaseModel):
    """构建索引请求体"""
    docs_dir: Optional[str] = Field(
        default=None,
        description="文档目录，不传则使用 config.yaml 中的配置",
    )
    force_rebuild: bool = Field(default=False, description="是否强制重建索引")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    index_loaded: bool
    redis_connected: bool
