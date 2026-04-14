from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from miniforensicsagent.tools import run_tool


class ToolsTest(unittest.TestCase):
    def test_read_uses_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
            result = run_tool(
                {"name": "Read", "arguments": {"file_path": "sample.txt", "offset": 2, "limit": 2}},
                root,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["content"], "b\nc")
            self.assertEqual(result["offset"], 2)
            self.assertEqual(result["limit"], 2)
            self.assertEqual(result["returned_lines"], 2)

    def test_grep_supports_alternation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.ts").write_text(
                "const sql = postgres(process.env.POSTGRES_URL!)\nconst fallback = process.env.DATABASE_URL\n",
                encoding="utf-8",
            )
            result = run_tool(
                {
                    "name": "Grep",
                    "arguments": {"pattern": "POSTGRES_URL|DATABASE_URL", "path": ".", "glob": "**/*.ts"},
                },
                root,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["matches"]), 2)
            self.assertIn("read_suggestions", result)
