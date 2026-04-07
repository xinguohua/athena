"""
ChatAnywhere 简易客户端（OpenAI Chat Completions 兼容）。

最简用法：
- 直接用 chatanywhere_summarize(text, api_key, endpoint)
- 或用 make_chatanywhere_summarizer(api_key, endpoint) 构建一个可复用的函数
"""
from __future__ import annotations

import json
from typing import Callable

import requests


def _default_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

# ------------------------------
# 极简版：直接传入 api_key 与 endpoint
# ------------------------------
def chatanywhere_summarize(
    text: str,
    *,
    api_key: str,
    endpoint: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    timeout: float = 30.0,
    return_usage: bool = False,
) -> str:
    if not text or not text.strip():
        return ("", {}) if return_usage else ""
    body = {
        "model": model,
        "temperature": max(0.0, float(temperature)),
        "messages": [
            {"role": "system", "content": "你是专业且简洁的安全技术总结器。"},
            {"role": "user", "content": text},
        ],
    }
    try:
        resp = requests.post(
            endpoint,
            headers=_default_headers(api_key),
            data=json.dumps(body),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
        result = str(content).strip()
        if not result:
            raise ValueError("LLM 返回空内容")
        if return_usage:
            usage = data.get("usage", {})
            return result, usage
        return result
    except Exception as ex:
        raise RuntimeError(f"LLM API 调用失败: {ex}") from ex


def make_chatanywhere_summarizer(
    *,
    api_key: str,
    endpoint: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> Callable[[str], str]:
    def _fn(text: str) -> str:
        return chatanywhere_summarize(
            text,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    return _fn
