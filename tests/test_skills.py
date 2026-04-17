from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from miniforensicsagent.skills import discover_skills, parse_skill_file, read_skill_resource


class SkillsTest(unittest.TestCase):
    def test_parse_skill_frontmatter_accepts_colons_in_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_file = Path(tmp) / "SKILL.md"
            skill_file.write_text(
                "---\n"
                "name: demo-skill\n"
                "description: Handles prompts: carefully and safely\n"
                "---\n"
                "\n"
                "# Demo\n",
                encoding="utf-8",
            )
            skill = parse_skill_file(skill_file)
            self.assertEqual(skill.name, "demo-skill")
            self.assertEqual(skill.description, "Handles prompts: carefully and safely")

    def test_discover_skills_prefers_native_project_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_skill = root / ".agents" / "skills" / "demo" / "SKILL.md"
            native_skill = root / ".mini-forensics-agent" / "skills" / "demo" / "SKILL.md"
            legacy_skill.parent.mkdir(parents=True, exist_ok=True)
            native_skill.parent.mkdir(parents=True, exist_ok=True)
            legacy_skill.write_text(
                "---\nname: demo\ndescription: legacy\n---\nLegacy body\n",
                encoding="utf-8",
            )
            native_skill.write_text(
                "---\nname: demo\ndescription: native\n---\nNative body\n",
                encoding="utf-8",
            )

            catalog = discover_skills(root)
            demo_skills = [skill for skill in catalog.skills if skill.name == "demo"]

            self.assertEqual(len(demo_skills), 1)
            self.assertEqual(demo_skills[0].description, "native")
            self.assertIn("overrides", "\n".join(catalog.diagnostics))

    def test_read_skill_resource_requires_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_file = root / ".mini-forensics-agent" / "skills" / "demo" / "SKILL.md"
            resource_file = skill_file.parent / "references" / "notes.md"
            resource_file.parent.mkdir(parents=True, exist_ok=True)
            skill_file.write_text(
                "---\nname: demo\ndescription: demo skill\n---\nUse references/notes.md\n",
                encoding="utf-8",
            )
            resource_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

            catalog = discover_skills(root)
            blocked = read_skill_resource(catalog, "demo", "references/notes.md", active_skill_names=set())
            allowed = read_skill_resource(catalog, "demo", "references/notes.md", offset=2, limit=2, active_skill_names={"demo"})

            self.assertFalse(blocked["ok"])
            self.assertTrue(allowed["ok"])
            self.assertEqual(allowed["content"], "line2\nline3")
