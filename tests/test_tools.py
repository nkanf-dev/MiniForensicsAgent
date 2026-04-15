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

    def test_read_accepts_end_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
            result = run_tool(
                {"name": "Read", "arguments": {"file_path": "sample.txt", "offset": 3, "limit": "end"}},
                root,
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["content"], "c\nd")
            self.assertEqual(result["offset"], 3)
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

    def test_glob_truncates_and_spills_when_too_many_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(90):
                (root / f"f{i:03d}.ts").write_text("export const x = 1;\n", encoding="utf-8")
            result = run_tool(
                {"name": "Glob", "arguments": {"pattern": "**/*.ts", "path": "."}},
                root,
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["truncated"])
            self.assertEqual(result["total_matches"], 90)
            self.assertEqual(len(result["matches"]), 80)
            self.assertIn("output_path", result)
            self.assertTrue((root / result["output_path"]).exists())

    def test_grep_truncates_and_spills_when_too_many_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = "\n".join(f"needle line {i}" for i in range(100)) + "\n"
            (root / "big.txt").write_text(content, encoding="utf-8")
            result = run_tool(
                {"name": "Grep", "arguments": {"pattern": "needle", "path": ".", "glob": "**/*.txt"}},
                root,
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["truncated"])
            self.assertEqual(result["total_matches"], 100)
            self.assertEqual(len(result["matches"]), 80)
            self.assertIn("output_path", result)
            self.assertTrue((root / result["output_path"]).exists())

    def test_turboquant_patch_optional(self) -> None:
        try:
            import turboquant_mlx  # noqa: F401
        except Exception:
            self.skipTest("turboquant-mlx not installed (optional extra)")

        from miniforensicsagent.models import patch_mlx_lm_prompt_cache_with_turboquant

        patch_mlx_lm_prompt_cache_with_turboquant(r_bits=4, theta_bits=4)

        import mlx_lm.models.cache as cache_module

        class DummyModel:
            layers = [object(), object(), object()]

        caches = cache_module.make_prompt_cache(DummyModel())
        self.assertEqual(len(caches), 3)
