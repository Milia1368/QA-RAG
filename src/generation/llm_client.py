"""
LLM 客户端封装
兼容 OpenAI API 格式（智谱GLM / vLLM / Qwen 均支持），支持流式与非流式输出。
"""

import json
from typing import Generator, Optional

import httpx
import requests
from loguru import logger
from openai import OpenAI


class LLMClient:
    """
    封装对智谱GLM、vLLM、Qwen系列OpenAI兼容接口调用，提供同步 / 流式两种接口。
    vLLM / 智谱大模型平台均兼容 OpenAI API，无需额外适配。
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model_name: str = "GLM-4.7-Flash",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 新建纯净 httpx 客户端；禁用 SDK 内置重试，避免与 Ollama 冷启动 502 叠加请求风暴
        clean_http_client = httpx.Client(timeout=180.0)
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            http_client=clean_http_client,
            max_retries=0,
        )
        logger.info(f"LLM 客户端初始化完成，模型: {model_name}, api_base: {api_base}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        非流式生成（返回完整字符串）。

        Args:
            prompt: 用户 prompt
            system_prompt: 系统提示词
            temperature: 采样温度，覆盖默认值
            max_tokens: 最大生成长度
            stream: 是否启用流式（此方法会收集流式输出后返回）

        Returns:
            模型生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stream=stream,
                )
                break
            except Exception as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status not in (502, 503, 504) or attempt == 2:
                    raise
                logger.warning(f"LLM 请求失败 ({status})，{attempt + 1}/3 次重试...")
                import time
                time.sleep(5 * (attempt + 1))
        else:
            raise last_err  # pragma: no cover

        if stream:
            full_text = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                full_text += delta
            return full_text

        return response.choices[0].message.content

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成（逐 token yield）。

        Yields:
            每次 LLM 输出的文本 delta
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
