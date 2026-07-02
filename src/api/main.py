"""
FastAPI 应用入口
负责：应用初始化、组件装配、生命周期管理。
"""
import os
import yaml
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.cache import RAGCache
from src.api.routers import router
from src.generation.factory import build_pipeline


def load_config(path: str = "configs/config.local.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期钩子，负责模型加载与资源释放。"""
    logger.info("=== RAG QA 系统启动中 ===")
    cfg = load_config()

    # ── 初始化缓存 ────────────────────────────────────
    redis_cfg = cfg["redis"]
    app.state.cache = RAGCache(
        host=redis_cfg["host"],
        port=redis_cfg["port"],
        db=redis_cfg["db"],
        password=redis_cfg.get("password", ""),
        ttl=redis_cfg["ttl"],
        max_connections=redis_cfg["max_connections"],
    )

    # ── 装配 RAG Pipeline ─────────────────────────────
    app.state.pipeline = build_pipeline(cfg)

    logger.info("=== 所有组件加载完成，服务就绪 ===")
    yield

    # 关闭时释放资源
    logger.info("=== RAG QA 系统关闭 ===")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG 企业知识库问答系统",
        description="基于 LangChain + FAISS + Qwen 的两阶段 RAG Pipeline，支持 Naive / HyDE / Adaptive HyDE",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = load_config("./configs/config.yaml")
    server_cfg = cfg["server"]
    uvicorn.run(
        "src.api.main:app",
        host=server_cfg["host"],
        port=server_cfg["port"],
        workers=server_cfg["workers"],
        timeout_keep_alive=server_cfg["timeout"],
        log_level="info",
    )
