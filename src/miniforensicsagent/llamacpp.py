from __future__ import annotations

import json
from typing import Any, Generator

import requests


class LlamaCPPHTTPClient:
    tokenizer = None
    last_usage: dict[str, Any] | None = None

    def __init__(self, base_url: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_chat(self, messages: list[dict[str, str]], config: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": getattr(config, "temperature", 0.3),
            "max_tokens": getattr(config, "max_tokens", 768),
            "top_p": getattr(config, "top_p", 0.9),
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [{}])
        content = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
        timings = data.get("timings", {})
        self.last_usage = {
            "prompt_tokens": timings.get("prompt_n", 0),
            "completion_tokens": timings.get("predicted_n", 0),
        }
        return {"text": content}

    def generate_stream_chat(self, messages: list[dict[str, str]], config: Any) -> Generator[str, None, None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": getattr(config, "temperature", 0.3),
            "max_tokens": getattr(config, "max_tokens", 768),
            "top_p": getattr(config, "top_p", 0.9),
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        last_timings: dict[str, Any] = {}
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if line.startswith("data: "):
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(raw)
                    choices = obj.get("choices", [{}])
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                    if choices[0].get("finish_reason"):
                        timings = obj.get("timings", {})
                        last_timings = timings
                        self.last_usage = {
                            "prompt_tokens": timings.get("prompt_n", 0),
                            "completion_tokens": timings.get("predicted_n", 0),
                        }
                except json.JSONDecodeError:
                    continue
        if self.last_usage is None and last_timings:
            self.last_usage = {
                "prompt_tokens": last_timings.get("prompt_n", 0),
                "completion_tokens": last_timings.get("predicted_n", 0),
            }