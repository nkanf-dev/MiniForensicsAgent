from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from miniforensicsagent.llamacpp import LlamaCPPHTTPClient


class TestLlamaCPPHTTPClient:
    def test_init_default_values(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        assert client.base_url == "http://localhost:8080"
        assert client.model == "model.gguf"
        assert client.timeout == 300
        assert client.use_chat is False
        assert client.last_usage is None

    def test_init_with_use_chat(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", use_chat=True)
        assert client.use_chat is True

    def test_generate_raises_not_implemented_for_chat(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", use_chat=True)
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9
        with pytest.raises(NotImplementedError):
            client.generate("prompt", config)

    def test_generate_chat_raises_not_implemented(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", use_chat=False)
        config = MagicMock()
        with pytest.raises(NotImplementedError):
            client.generate_chat("prompt", config)

    def test_generate_stream_raises_not_implemented_for_chat(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", use_chat=True)
        config = MagicMock()
        with pytest.raises(NotImplementedError):
            list(client.generate_stream("prompt", config))

    def test_generate_stream_chat_raises_not_implemented(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", use_chat=False)
        config = MagicMock()
        with pytest.raises(NotImplementedError):
            list(client.generate_stream_chat("prompt", config))