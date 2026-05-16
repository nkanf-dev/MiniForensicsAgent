from __future__ import annotations

import json
from typing import Any, Generator

import requests


class LlamaCPPHTTPClient:
    tokenizer = None

    def __init__(self, base_url: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _build_payload(self, prompt: str, config: Any, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "temperature": getattr(config, "temperature", 0.3),
            "max_tokens": getattr(config, "max_tokens", 768),
            "top_p": getattr(config, "top_p", 0.9),
        }
        stop_seq = getattr(config, "stop_sequences", None)
        if stop_seq:
            payload["stop"] = stop_seq
        return payload

    def generate(self, prompt: str, config: Any) -> dict[str, Any]:
        payload = self._build_payload(prompt, config, stream=False)
        resp = requests.post(
            f"{self.base_url}/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["text"]
        return {"text": content}

    def generate_stream(self, prompt: str, config: Any) -> Generator[str, None, None]:
        payload = self._build_payload(prompt, config, stream=True)
        resp = requests.post(
            f"{self.base_url}/completions",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        stop_sequences: list[str] = getattr(config, "stop_sequences", []) or []
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    return
                try:
                    obj = json.loads(raw)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue
                    for stop_seq in stop_sequences:
                        if stop_seq in content:
                            yield content[: content.index(stop_seq) + len(stop_seq)]
                            return
                    yield content
                except json.JSONDecodeError:
                    continue