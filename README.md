# RAG 企业知识库问答系统

基于检索增强生成（Retrieval-Augmented Generation）的内部文档 QA 系统，支持 Naive RAG 与 HyDE 双路检索策略，提供 FastAPI 服务化部署。

## 技术栈

| 层级     | 技术选型                        |
| -------- | ------------------------------- |
| 文档解析 | LangChain Document Loaders      |
| 向量化   | `BAAI/bge-large-zh-v1.5`      |
| 向量存储 | FAISS                           |
| LLM      | Qwen2.5-7B-Instruct (vLLM 推理) |
| 服务框架 | FastAPI + SSE 流式输出          |
| 缓存     | Redis                           |
| 评估     | RAGAS (Faithfulness / MRR@5)    |

## 项目结构

```
rag-qa-system/
├── src/
│   ├── ingestion/          # 文档处理 & 向量化索引
│   │   ├── __init__.py
│   │   ├── loader.py       # 多格式文档加载
│   │   ├── chunker.py      # Chunk 切分策略（512 token + 64 overlap）
│   │   └── indexer.py      # FAISS 索引构建
│   ├── retrieval/          # 检索模块
│   │   ├── __init__.py
│   │   ├── naive_rag.py    # 标准向量检索
│   │   ├── hyde.py         # HyDE 假设文档嵌入
│   │   └── reranker.py     # BGE Reranker 重排序
│   ├── generation/         # 生成模块
│   │   ├── __init__.py
│   │   ├── llm_client.py   # Qwen LLM 封装
│   │   ├── prompt.py       # Prompt 模板
│   │   └── pipeline.py     # RAG Pipeline 主流程
│   ├── api/                # FastAPI 服务
│   │   ├── __init__.py
│   │   ├── main.py         # 应用入口
│   │   ├── routers.py      # 路由定义
│   │   ├── schemas.py      # 请求/响应模型
│   │   └── cache.py        # Redis 缓存层
│   └── evaluation/         # 评估模块
│       ├── __init__.py
│       ├── metrics.py      # Faithfulness / MRR@5
│       └── compare.py      # Naive / HyDE / Adaptive HyDE 对比实验
├── docs/
│   └── eval_analysis.md    # 实验评测分析与改进方案
├── configs/
│   └── config.yaml         # 全局配置
├── data/
│   └── sample_docs/        # 示例文档
├── scripts/
│   ├── build_index.sh      # 一键建索引
│   ├── prepare_eval.sh     # 构建评测集
│   ├── sample_eval.sh      # 随机抽取评测子集
│   ├── run_eval.sh         # 三路对比评测
│   └── start_server.sh     # 启动服务
├── tests/
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# Homebrew 安装 brew install redis # 后台启动 redis 服务 
brew services start redis # 验证是否启动成功 
redis-cli ping # 返回 PONG 代表正常，默认地址 localhost:6379，无密码
```

### 2. 修改配置

```bash
cp configs/config.yaml configs/config.local.yaml
# 编辑 config.local.yaml，填入模型路径与 Redis 地址
```

### 3. 构建向量索引

```bash
bash scripts/build_index.sh  data/sample_docs
```

### 4. 启动服务

```bash
bash scripts/start_server.sh
# 或 Docker 一键启动
docker-compose up -d
```

### 5. 测试接口

```bash
# 普通问答
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "公司差旅报销流程是什么？", "mode": "hyde"}'

# 流式输出
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "研发规范文档在哪里？", "mode": "naive"}'
```

### 后续处理

#### 1.停止Redis缓存

```
brew services stop redis
```

#### 2.重建索引

```
rm -rf data/faiss\_index 
bash scripts/build\_index.sh data/sample\_docs
```

#### 3.关闭服务

```
# Ctrl + C 终止uvicorn进程
```

#### 4.清理模型缓存

```
rm -rf \~/.cache/huggingface
```

## 核心特性

- **两阶段 RAG Pipeline**：Chunk → 向量化 → 检索 → Rerank → 生成
- **HyDE 检索**：先生成假设文档再向量化查询，提升歧义 Query 的 MRR@5
- **答案溯源**：返回答案来源文档段落，降低幻觉率
- **流式输出**：SSE 实时推送，P99 延迟 < 800ms
- **Redis 缓存**：相同 Query 命中缓存，避免重复推理

## 实验评测

在 **QuestAnswer1Doc** 中文单文档 QA 语料（2000 篇新闻 `.txt`）上，对 Naive / HyDE / Adaptive HyDE / **Hybrid** 四路检索进行对比评测。Hard 子集（Naive@5 未命中）约 **48/200（24%）**。

### 快速运行

```bash
# 构建评测集 & 随机抽 200 条
bash scripts/prepare_eval.sh
bash scripts/sample_eval.sh 200

# 导出 Hard 子集（Naive 未命中 @5，无需 Ollama）
bash scripts/extract_hard_subset.sh data/eval_dataset_200.jsonl

# 四路对比 naive / hyde / adaptive_hyde / hybrid（HyDE 需 Ollama）
EVAL_DATA=data/eval_dataset_200.jsonl \
OUTPUT=data/eval_report_200.json \
bash scripts/run_eval.sh

# Hard 子集上快速验证（⚠️ Naive MRR 恒为 0，详见 docs/eval_analysis.md §10）
EVAL_DATA=data/eval_dataset_200_hard.jsonl \
bash scripts/run_eval.sh --retrieval_only --modes hyde,adaptive_hyde,hybrid
```

默认服务检索模式为 **`hybrid`**（Naive ∪ HyDE 并集 + Adaptive 门控）。详细分析与改进说明见 **[docs/eval_analysis.md](docs/eval_analysis.md)**。

### 最新结果（200 条子集，改进前基线）
|------|-------|----------|-----------|----------|
| **Naive** | **0.643** | **0.760** | **0.515** | 3.2s |
| HyDE | 0.527 | 0.625 | 0.450 | 4.8s |
| Adaptive HyDE | 0.548 | 0.660 | 0.460 | 5.0s |

Hard 子集（48 条，Naive@5 未命中）上 HyDE 仅救回 **2/48**，MRR≈0.04；Naive 在该子集上 **MRR 恒为 0**（筛选定义所致）。解读见 [docs/eval_analysis.md §10](docs/eval_analysis.md#10-hard-子集报告解读eval_report_200_hardjson)。

> 完整报告见 `data/eval_report_200.json`。Naive 在本数据集上优于 HyDE 的原因分析、Hard 子集评测与后续改进路线，见 **[docs/eval_analysis.md](docs/eval_analysis.md)**。


