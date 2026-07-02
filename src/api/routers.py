"""
FastAPI 路由定义
- POST /api/v1/chat         非流式问答
- GET  /api/v1/chat/stream  SSE 流式问答
- POST /api/v1/index/build  触发索引构建
- GET  /api/v1/health       健康检查
"""

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexBuildRequest,
    SourceDoc,
)

router = APIRouter(prefix="/api/v1")


def get_pipeline(request: Request):
    """从 app.state 获取 RAGPipeline 实例（依赖注入）。"""
    return request.app.state.pipeline


def get_cache(request: Request):
    """从 app.state 获取 RAGCache 实例（依赖注入）。"""
    return request.app.state.cache


# ── 非流式问答 ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="知识库问答（非流式）")
async def chat(
    body: ChatRequest,
    pipeline=Depends(get_pipeline),
    cache=Depends(get_cache),
):
    """
    标准问答接口，返回完整答案与溯源文档。
    相同问题命中 Redis 缓存时直接返回，跳过推理。
    """
    t0 = time.perf_counter()

    # 尝试命中缓存
    cached_result = cache.get(body.question, body.mode)
    if cached_result:
        cached_result["cached"] = True
        logger.info(f"[API] 缓存命中，耗时 {(time.perf_counter()-t0)*1000:.1f}ms")
        return ChatResponse(**cached_result)

    # 执行 RAG Pipeline
    try:
        result = pipeline.run(query=body.question, mode=body.mode)
    except Exception as e:
        logger.error(f"[API] Pipeline 执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response_dict = {
        "answer": result.answer,
        "sources": result.sources,
        "retrieval_mode": result.retrieval_mode,
        "query": result.query,
        "cached": False,
    }

    # 写入缓存
    cache.set(body.question, body.mode, response_dict)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[API] 问答完成，耗时 {elapsed_ms:.1f}ms")

    return ChatResponse(**response_dict)


# ── SSE 流式问答 ────────────────────────────────────────────────────────────

@router.post("/chat/stream", summary="知识库问答（SSE 流式输出）")
async def chat_stream(
    body: ChatRequest,
    pipeline=Depends(get_pipeline),
):
    """
    SSE 流式问答接口。
    - 首帧推送 [SOURCES] 来源信息
    - 后续帧逐 token 推送答案
    - 最终帧发送 [DONE] 信号

    Content-Type: text/event-stream
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in pipeline.stream_run(query=body.question, mode=body.mode):
                if chunk.startswith("[SOURCES]"):
                    # 来源信息帧
                    yield f"event: sources\ndata: {chunk}\n\n"
                else:
                    # 答案文本帧
                    yield f"event: token\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[API/Stream] 生成异常: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # 禁用 Nginx 缓冲，保证实时推送
        },
    )


# ── 健康检查 ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(request: Request, cache=Depends(get_cache)):
    pipeline = getattr(request.app.state, "pipeline", None)
    return HealthResponse(
        status="ok",
        index_loaded=pipeline is not None,
        redis_connected=cache.is_available,
    )
