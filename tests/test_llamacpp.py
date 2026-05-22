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
        assert client.last_usage is None

    def test_init_with_timeout(self) -> None:
        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf", timeout=600)
        assert client.timeout == 600