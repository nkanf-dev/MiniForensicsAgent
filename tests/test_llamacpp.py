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


class TestGenerateChat:
    @patch("miniforensicsagent.llamacpp.requests.post")
    def test_generate_chat_builds_payload(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "timings": {"prompt_n": 100, "predicted_n": 20},
        }
        mock_post.return_value = mock_response

        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9

        messages = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}]
        result = client.generate_chat(messages, config)

        assert result["text"] == "done"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == "model.gguf"
        assert call_kwargs["json"]["messages"] == messages
        assert call_kwargs["json"]["stream"] is False

    @patch("miniforensicsagent.llamacpp.requests.post")
    def test_generate_chat_updates_last_usage(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}],
            "timings": {"prompt_n": 50, "predicted_n": 10},
        }
        mock_post.return_value = mock_response

        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9

        client.generate_chat([], config)
        assert client.last_usage == {"prompt_tokens": 50, "completion_tokens": 10}


class TestGenerateStreamChat:
    @patch("miniforensicsagent.llamacpp.requests.post")
    def test_generate_stream_chat_yields_content(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter([
            b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
            b"data: [DONE]",
        ])
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9

        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(client.generate_stream_chat(messages, config))

        assert chunks == ["hello", " world"]

    @patch("miniforensicsagent.llamacpp.requests.post")
    def test_generate_stream_chat_updates_last_usage(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter([
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"timings":{"prompt_n":30,"predicted_n":5}}',
            b"data: [DONE]",
        ])
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9

        list(client.generate_stream_chat([], config))
        assert client.last_usage == {"prompt_tokens": 30, "completion_tokens": 5}

    @patch("miniforensicsagent.llamacpp.requests.post")
    def test_generate_stream_chat_uses_chat_endpoint(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter([b"data: [DONE]"])
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = LlamaCPPHTTPClient("http://localhost:8080", "model.gguf")
        config = MagicMock()
        config.temperature = 0.3
        config.max_tokens = 768
        config.top_p = 0.9

        list(client.generate_stream_chat([], config))
        url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert "/chat/completions" in url