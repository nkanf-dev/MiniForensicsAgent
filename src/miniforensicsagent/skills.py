from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable
import re


DEFAULT_SKILL_READ_LIMIT = 120
MAX_SKILL_RESOURCE_PREVIEW = 24


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    skill_file: Path
    metadata: dict[str, str]
    content: str

    @property
    def skill_dir(self) -> Path:
        return self.skill_file.parent


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[SkillRecord, ...]
    diagnostics: tuple[str, ...]
    roots: tuple[Path, ...]

    def by_name(self) -> dict[str, SkillRecord]:
        return {skill.name: skill for skill in self.skills}


def candidate_skill_roots(project_root: Path, extra_dirs: Iterable[str] = ()) -> tuple[Path, ...]:
    raw_roots = [
        Path.home() / ".agents" / "skills",
        Path.home() / ".mini-forensics-agent" / "skills",
        project_root / ".agents" / "skills",
        project_root / ".mini-forensics-agent" / "skills",
        *(Path(raw).expanduser() for raw in extra_dirs),
    ]
    seen: set[Path] = set()
    resolved: list[Path] = []
    for root in raw_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        resolved.append(root)
    return tuple(resolved)


def discover_skills(project_root: Path, extra_dirs: Iterable[str] = ()) -> SkillCatalog:
    skills_by_name: dict[str, SkillRecord] = {}
    diagnostics: list[str] = []
    roots = candidate_skill_roots(project_root, extra_dirs=extra_dirs)
    for root in roots:
        if not root.exists():
            continue
        for skill_file in sorted(root.rglob("SKILL.md")):
            try:
                skill = parse_skill_file(skill_file)
            except Exception as exc:
                diagnostics.append(f"Failed to parse skill {skill_file}: {exc}")
                continue
            previous = skills_by_name.get(skill.name)
            if previous is not None and previous.skill_file != skill.skill_file:
                diagnostics.append(
                    f"Skill {skill.name!r} from {skill.skill_file} overrides {previous.skill_file}"
                )
            skills_by_name[skill.name] = skill
    return SkillCatalog(
        skills=tuple(sorted(skills_by_name.values(), key=lambda item: item.name.lower())),
        diagnostics=tuple(diagnostics),
        roots=roots,
    )


def parse_skill_file(skill_file: Path) -> SkillRecord:
    raw = skill_file.read_text(encoding="utf-8")
    match = re.match(r"(?s)\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", raw)
    if match is None:
        raise ValueError("missing YAML frontmatter")
    metadata = _parse_frontmatter(match.group(1))
    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    if not name:
        raise ValueError("frontmatter is missing name")
    if not description:
        raise ValueError("frontmatter is missing description")
    return SkillRecord(
        name=name,
        description=description,
        skill_file=skill_file.resolve(),
        metadata=metadata,
        content=match.group(2).strip(),
    )


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    lines = frontmatter.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {"|", ">"}:
            block: list[str] = []
            while index < len(lines):
                next_line = lines[index]
                if next_line.strip() and not next_line.startswith((" ", "\t")):
                    break
                block.append(next_line.lstrip())
                index += 1
            metadata[key] = "\n".join(block).strip()
            continue
        metadata[key] = _strip_yaml_scalar(value)
    return metadata


def _strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def render_skill_catalog(catalog: SkillCatalog) -> str:
    if not catalog.skills:
        return ""
    lines = ["<available_skills>"]
    for skill in catalog.skills:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(skill.description)}</description>",
                f"    <location>{escape(str(skill.skill_file))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def render_active_skill_context(active_skills: dict[str, dict[str, Any]]) -> str:
    if not active_skills:
        return ""
    sections = ["Activated skills:"]
    for name in sorted(active_skills):
        skill = active_skills[name]
        sections.extend(
            [
                f"<active_skill name={name!r}>",
                f"description: {skill['description']}",
                f"location: {skill['location']}",
            ]
        )
        resources = skill.get("resource_files", [])
        if resources:
            sections.append("resource_files:")
            sections.extend(f"- {item}" for item in resources)
        sections.append("instructions:")
        sections.append(str(skill.get("content", "")).strip())
        sections.append("</active_skill>")
    return "\n".join(sections)


def activate_skill(catalog: SkillCatalog, skill_name: str) -> dict[str, Any]:
    skill = catalog.by_name().get(skill_name)
    if skill is None:
        return {"ok": False, "error": f"Unknown skill: {skill_name}"}
    resource_files = list_skill_resources(skill.skill_dir)
    payload: dict[str, Any] = {
        "ok": True,
        "name": skill.name,
        "description": skill.description,
        "location": str(skill.skill_file),
        "content": skill.content,
        "resource_files": resource_files,
    }
    for field in ("allowed-tools", "compatibility", "version"):
        if skill.metadata.get(field):
            payload[field.replace("-", "_")] = skill.metadata[field]
    return payload


def list_skill_resources(skill_dir: Path) -> list[str]:
    resources: list[str] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SKILL.md":
            continue
        try:
            rel = str(path.relative_to(skill_dir))
        except ValueError:
            continue
        resources.append(rel)
        if len(resources) >= MAX_SKILL_RESOURCE_PREVIEW:
            break
    return resources


def read_skill_resource(
    catalog: SkillCatalog,
    skill_name: str,
    relative_path: str,
    *,
    offset: int = 1,
    limit: int | str = DEFAULT_SKILL_READ_LIMIT,
    active_skill_names: set[str] | None = None,
) -> dict[str, Any]:
    if active_skill_names is not None and skill_name not in active_skill_names:
        return {"ok": False, "error": f"Skill is not active: {skill_name}"}
    skill = catalog.by_name().get(skill_name)
    if skill is None:
        return {"ok": False, "error": f"Unknown skill: {skill_name}"}
    candidate = (skill.skill_dir / relative_path).resolve()
    if candidate != skill.skill_dir and skill.skill_dir not in candidate.parents:
        return {"ok": False, "error": f"Path escapes skill directory: {relative_path}"}
    if not candidate.exists() or not candidate.is_file():
        return {"ok": False, "error": f"Skill resource not found: {relative_path}"}
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return {"ok": False, "error": f"Skill resource is not UTF-8 text: {relative_path}"}
    start = max(1, int(offset))
    start_index = start - 1
    if str(limit).strip().lower() in {"end", "eof", "-1"}:
        parsed_limit = max(1, len(lines) - start_index)
        end_index = len(lines)
    else:
        parsed_limit = max(1, int(limit))
        end_index = start_index + parsed_limit
    chunk = lines[start_index:end_index]
    return {
        "ok": True,
        "skill_name": skill.name,
        "file_path": relative_path,
        "content": "\n".join(chunk),
        "offset": start,
        "limit": parsed_limit,
        "returned_lines": len(chunk),
        "total_lines": len(lines),
        "truncated": end_index < len(lines),
    }
