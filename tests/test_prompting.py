from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from miniforensicsagent.prompting import build_chat_messages


class TestBuildChatMessages:
    def test_first_turn_system_and_user(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            messages = build_chat_messages(
                task="Find the config",
                transcript=[],
                workspace=workspace,
            )
            assert len(messages) >= 2
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "user"
            assert "Find the config" in messages[-1]["content"]

    def test_single_tool_call(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            transcript = [
                {
                    "raw": '<tool_call>{"name":"Glob","arguments":{"pattern":"**/*.py","path":"."}}</tool_call>',
                    "decision": {"type": "tool", "name": "Glob", "arguments": {"pattern": "**/*.py", "path": "."}},
                    "observations": [
                        {
                            "ok": True,
                            "matches": ["main.py", "utils.py"],
                        }
                    ],
                }
            ]
            messages = build_chat_messages(
                task="Find config",
                transcript=transcript,
                workspace=workspace,
            )
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            assert len(assistant_msgs) >= 1
            assert "Glob" in assistant_msgs[0]["content"]

    def test_observation_truncation(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            long_content = "x" * 3000
            transcript = [
                {
                    "raw": '<tool_call>{"name":"Read","arguments":{"file_path":"big.txt"}}</tool_call>',
                    "decision": {"type": "tool", "name": "Read", "arguments": {"file_path": "big.txt"}},
                    "observations": [{"ok": True, "content": long_content}],
                }
            ]
            messages = build_chat_messages(
                task="Read file",
                transcript=transcript,
                workspace=workspace,
            )
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            assert len(tool_msgs) == 1
            assert len(tool_msgs[0]["content"]) <= 2000 + 50
            assert "... [truncated" in tool_msgs[0]["content"]

    def test_plan_in_transcript(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            transcript = [
                {
                    "raw": '{"type":"plan","goal":"find config","steps":["search"],"done_when":"found"}',
                    "decision": {"type": "plan", "plan": {"goal": "find config", "steps": ["search"], "done_when": "found"}},
                    "observations": [{"ok": True, "plan_captured": True}],
                }
            ]
            messages = build_chat_messages(
                task="Find config",
                transcript=transcript,
                workspace=workspace,
            )
            assert len(messages) >= 3

    def test_final_in_transcript(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            transcript = [
                {
                    "raw": "<final>done</final>",
                    "decision": {"type": "final", "answer": "done"},
                    "observations": [],
                }
            ]
            messages = build_chat_messages(
                task="Find config",
                transcript=transcript,
                workspace=workspace,
            )
            assistant_msgs = [m for m in messages if m["role"] == "assistant"]
            assert len(assistant_msgs) >= 1

    def test_current_plan_in_user_message(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            messages = build_chat_messages(
                task="Find config",
                transcript=[],
                workspace=workspace,
                current_plan={"goal": "find it", "steps": ["search"], "done_when": "found"},
            )
            user_content = messages[-1]["content"]
            assert "Current plan" in user_content

    def test_remaining_iterations_warning(self) -> None:
        with tempfile.TemporaryDirectory():
            workspace = Path(".")
            messages = build_chat_messages(
                task="Find config",
                transcript=[],
                workspace=workspace,
                remaining_iterations=1,
            )
            user_content = messages[-1]["content"]
            assert "1" in user_content or "iteration" in user_content.lower()