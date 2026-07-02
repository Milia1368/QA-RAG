from string import Template
from typing import List

# 清洗弯引号，解决ascii编码报错
def clean_quotes(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    return text

# ── RAG QA 主模板 ─────────────────────────────────────────────────────────────
RAG_QA_SYSTEM = """你是一位专业的企业内部知识库助手。
你的任务是根据提供的文档上下文，准确、简洁地回答用户问题。

回答规范：
1. 仅依据上下文中的信息作答，不要编造或推测未提及的内容。
2. 如果上下文中没有相关信息，明确告知用户"根据现有文档，无法找到该问题的答案"。
3. 回答时引用具体来源（如文档名称、章节），标注对应[文档X]编号，便于用户溯源核实。
4. 保持简洁，优先给出核心结论，再补充细节。"""

RAG_QA_USER_TEMPLATE = Template("""请根据以下参考文档回答用户问题。

【参考文档】
$context

【用户问题】
$question

硬性规则：回答里所有引用内容必须标注对应[文档数字]；没有匹配资料直接回复找不到答案。
【回答】""")


def build_context(docs: List, max_total_chars: int = 3000) -> str:
    """
    将检索到的 Document 列表拼接为上下文字符串，附带来源标注，限制总长度。
    """
    parts = []
    total_len = 0
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        content_text = clean_quotes(doc.page_content.strip())
        block = f"[文档{i}] 来源: {source} | 第{chunk_idx}段\n内容: {content_text}"

        if total_len + len(block) > max_total_chars:
            break
        parts.append(block)
        total_len += len(block) + 2

    return "\n\n".join(parts)


def build_rag_prompt(question: str, docs: List) -> tuple[str, str]:
    """
    构造 RAG 问答的 system prompt 和 user prompt。
    """
    # 限制问题长度
    question = clean_quotes(question[:800])
    context = build_context(docs, max_total_chars=3000)
    # 空文档兜底
    if not context.strip():
        context = "无任何相关参考文档。"

    # safe_substitute 避免文本含$导致模板报错
    user_prompt = RAG_QA_USER_TEMPLATE.safe_substitute(
        context=context,
        question=question,
    )
    return RAG_QA_SYSTEM, user_prompt
