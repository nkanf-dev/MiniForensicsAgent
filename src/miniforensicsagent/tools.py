from __future__ import annotations

import fnmatch
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .skills import DEFAULT_SKILL_READ_LIMIT, SkillCatalog, activate_skill, read_skill_resource

IS_WINDOWS = sys.platform == "win32"
_has_rg_cache: bool | None = None


def _has_rg() -> bool:
    global _has_rg_cache
    if _has_rg_cache is None:
        _has_rg_cache = shutil.which("rg") is not None
    return _has_rg_cache


BASH_TRANSLATIONS: dict[str, str] = {
    "pwd": "cd",
    "ls": "dir /b" if IS_WINDOWS else "ls",
    "ls -la": "dir /a" if IS_WINDOWS else "ls -la",
    "cat": "type",
    "find": "dir /s /b" if IS_WINDOWS else "find",
    "which": "where" if IS_WINDOWS else "which",
    "cp": "copy" if IS_WINDOWS else "cp",
    "mv": "move" if IS_WINDOWS else "mv",
    "rm": "del /f" if IS_WINDOWS else "rm",
    "mkdir": "mkdir",
    "rmdir": "rmdir",
    "clear": "cls" if IS_WINDOWS else "clear",
    "echo": "echo",
    "head": "more +1" if IS_WINDOWS else "head",
    "tail": "powershell -c \"Get-Content file | Select-Object -Last N\"" if IS_WINDOWS else "tail",
    "wc": "findstr /r" if IS_WINDOWS else "wc",
}


def _translate_bash_command(command: str) -> str:
    if not IS_WINDOWS:
        return command
    original = command
    command = command.strip()
    parts = shlex.split(command)
    if not parts:
        return original
    base = parts[0].lower()
    translated = BASH_TRANSLATIONS.get(base)
    if translated:
        parts[0] = translated
        return " ".join(parts)
    return original


DEFAULT_READ_LIMIT = 80
MAX_MATCH_PREVIEW = 80
MAX_MATCH_TEXT_PREVIEW = 240
SPILL_DIRNAME = ".mini_forensics_spill"


def expand_brace_pattern(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(expand_brace_pattern(pattern[: match.start()] + option + pattern[match.end() :]))
    return expanded


def run_tool(
    call: dict[str, Any],
    workspace: Path,
    *,
    skill_catalog: SkillCatalog | None = None,
    active_skill_names: set[str] | None = None,
) -> dict[str, Any]:
    tool = call["name"]
    args = call["arguments"]
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    def resolve_inside_workspace(raw_path: str) -> Path:
        if raw_path.strip() in {"", "/"}:
            raw_path = "."
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            if path.parts and path.parts[0] == workspace.name:
                path = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(".")
            path = workspace / path
        resolved = path.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"Path outside workspace: {resolved}")
        return resolved

    def nearby_listing(path: Path) -> list[str]:
        base = path if path.exists() and path.is_dir() else path.parent
        if not base.exists() or not base.is_dir():
            base = workspace
        try:
            return sorted(item.name for item in base.iterdir())[:20]
        except Exception:
            return []

    def suggest_related_files(root: Path, pattern: str, glob_patterns: list[str]) -> list[str]:
        terms = [part.lower() for part in re.split(r"[^A-Za-z0-9_]+", pattern) if part and len(part) >= 3]
        related: list[str] = []
        search_root = root if root.exists() else workspace
        for item in search_root.rglob("*"):
            if not item.is_file():
                continue
            rel_workspace = str(item.relative_to(workspace))
            rel_root = str(item.relative_to(search_root))
            if not (any(fnmatch.fnmatch(item.name, g) for g in glob_patterns) or any(fnmatch.fnmatch(rel_root, g) for g in glob_patterns) or any(fnmatch.fnmatch(rel_workspace, g) for g in glob_patterns)):
                continue
            haystacks = [item.name.lower(), rel_workspace.lower()]
            if any(term in hay for term in terms for hay in haystacks):
                related.append(rel_workspace)
            if len(related) >= 8:
                break
        return related

    def build_read_suggestions(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for match in matches:
            file_path = str(match.get("file", ""))
            offset = max(1, int(match.get("line", 1)) - 10)
            key = (file_path, offset)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append({"file_path": file_path, "offset": offset, "limit": 30})
            if len(suggestions) >= 5:
                break
        return suggestions

    def suggest_related_paths_for_glob(root: Path, pattern: str) -> list[str]:
        terms = [part.lower() for part in re.split(r"[^A-Za-z0-9_]+", pattern.replace("*", " ")) if part and len(part) >= 2]
        search_root = root if root.exists() else workspace
        related: list[str] = []
        for item in search_root.rglob("*"):
            haystack = str(item.relative_to(workspace)).lower()
            if any(term in haystack for term in terms):
                related.append(str(item.relative_to(workspace)))
            if len(related) >= 8:
                break
        return related

    def trim_text_for_preview(text: str, max_chars: int = MAX_MATCH_TEXT_PREVIEW) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    def write_spill(tool_name: str, payload: dict[str, Any]) -> str:
        spill_dir = workspace / SPILL_DIRNAME
        spill_dir.mkdir(parents=True, exist_ok=True)
        spill_file = spill_dir / f"{tool_name.lower()}_{int(time.time() * 1000)}.json"
        spill_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(spill_file.relative_to(workspace))

    def parse_positive_int(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return max(1, value)
        if isinstance(value, float):
            return max(1, int(value))
        try:
            parsed = int(str(value).strip())
            return max(1, parsed)
        except Exception:
            return default

    try:
        if tool == "ActivateSkill":
            if skill_catalog is None:
                return {"ok": False, "error": "Skills are not enabled for this run."}
            return activate_skill(skill_catalog, str(args.get("skill_name", "")))

        if tool == "ReadSkillResource":
            if skill_catalog is None:
                return {"ok": False, "error": "Skills are not enabled for this run."}
            return read_skill_resource(
                skill_catalog,
                str(args.get("skill_name", "")),
                str(args.get("file_path", "")),
                offset=args.get("offset", 1),
                limit=args.get("limit", DEFAULT_SKILL_READ_LIMIT),
                active_skill_names=active_skill_names,
            )

        if tool == "Read":
            file_path = resolve_inside_workspace(str(args["file_path"]))
            offset = parse_positive_int(args.get("offset", args.get("start_line", 1)), 1)
            raw_limit = args.get("limit", args.get("max_lines", DEFAULT_READ_LIMIT))
            lines = file_path.read_text(encoding="utf-8").splitlines()
            start_index = offset - 1
            if str(raw_limit).strip().lower() in {"end", "eof", "-1"}:
                limit = max(1, len(lines) - start_index)
                end_index = len(lines)
            else:
                limit = parse_positive_int(raw_limit, DEFAULT_READ_LIMIT)
                end_index = start_index + limit
            chunk = lines[start_index:end_index]
            return {"ok": True, "content": "\n".join(chunk), "offset": offset, "limit": limit, "returned_lines": len(chunk), "total_lines": len(lines), "truncated": end_index < len(lines)}

        if tool == "Glob":
            root = resolve_inside_workspace(str(args.get("path", ".")))
            if not root.exists():
                return {"ok": False, "error": f"Path does not exist: {root}", "cwd_hint": str((root.parent if root.parent.exists() else workspace).resolve()), "entries": nearby_listing(root)}
            patterns = [item.replace(".*", "*") for item in expand_brace_pattern(str(args["pattern"]))]

            if _has_rg():
                matches: list[str] = []
                for pattern in patterns:
                    completed = subprocess.run(["rg", "--files", str(root), "-g", pattern], cwd=workspace, capture_output=True, text=True, timeout=180, check=False)
                    for line in completed.stdout.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            matches.append(str(Path(line).resolve().relative_to(workspace)))
                        except Exception:
                            continue
                matches = sorted(dict.fromkeys(matches))
            else:
                matches = []
                for pattern in patterns:
                    base_pattern = pattern.lstrip("/\\")
                    for item in root.rglob(base_pattern):
                        if item.is_file():
                            try:
                                matches.append(str(item.relative_to(workspace)))
                            except ValueError:
                                continue
                matches = sorted(dict.fromkeys(matches))
            if not matches:
                related = suggest_related_paths_for_glob(root, str(args["pattern"]))
                hint = "No glob matches. Try a broader pattern, inspect nearby directories, or search a parent path first."
                if related:
                    hint += f" Related paths: {', '.join(related[:5])}"
                return {"ok": True, "matches": [], "hint": hint, "related_files": related[:8]}
            if len(matches) > MAX_MATCH_PREVIEW:
                preview = matches[:MAX_MATCH_PREVIEW]
                spill_path = write_spill(
                    "Glob",
                    {
                        "tool": "Glob",
                        "workspace": str(workspace),
                        "path": str(args.get("path", ".")),
                        "pattern": str(args.get("pattern", "")),
                        "total_matches": len(matches),
                        "matches": matches,
                    },
                )
                return {
                    "ok": True,
                    "matches": preview,
                    "truncated": True,
                    "total_matches": len(matches),
                    "omitted_matches": len(matches) - len(preview),
                    "output_path": spill_path,
                    "hint": "Result list is long. Showing preview only; full matches were spilled to output_path.",
                }
            return {"ok": True, "matches": matches}

        if tool == "Grep":
            root = resolve_inside_workspace(str(args.get("path", ".")))
            if not root.exists():
                return {"ok": False, "error": f"Path does not exist: {root}", "cwd_hint": str((root.parent if root.parent.exists() else workspace).resolve()), "entries": nearby_listing(root)}
            pattern = str(args["pattern"])
            glob_patterns = expand_brace_pattern(str(args.get("glob", "*")))

            def _parse_grep_result(rel_path: str, line_no: int, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
                full = {"file": rel_path, "line": line_no, "text": text}
                preview_text = trim_text_for_preview(text)
                preview = {"file": rel_path, "line": line_no, "text": preview_text}
                return full, preview

            matches: list[dict[str, Any]] = []
            full_matches: list[dict[str, Any]] = []
            trimmed_lines = 0

            if _has_rg():
                rg_cmd = ["rg", "-n", "-S", "--no-heading", pattern]
                for glob_pattern in glob_patterns:
                    rg_cmd.extend(["-g", glob_pattern])
                rg_cmd.append(str(root))
                completed = subprocess.run(rg_cmd, cwd=workspace, capture_output=True, text=True, timeout=180, check=False)
                for line in completed.stdout.splitlines():
                    parts = line.split(":", 2)
                    if len(parts) != 3:
                        continue
                    file_part, line_no, text = parts
                    try:
                        rel = str(Path(file_part).resolve().relative_to(workspace))
                    except Exception:
                        rel = file_part
                    line_value = int(line_no) if line_no.isdigit() else 1
                    full, preview = _parse_grep_result(rel, line_value, text)
                    full_matches.append(full)
                    if preview["text"] != text:
                        trimmed_lines += 1
                    matches.append(preview)
            else:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    return {"ok": False, "error": f"Invalid regex pattern: {pattern}"}
                for item in root.rglob("*"):
                    if not item.is_file():
                        continue
                    rel = str(item.relative_to(root))
                    if not any(fnmatch.fnmatch(rel, g) for g in glob_patterns):
                        continue
                    try:
                        content = item.read_text(encoding="utf-8", errors="replace")
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                full, preview = _parse_grep_result(rel, i, line)
                                full_matches.append(full)
                                if preview["text"] != line:
                                    trimmed_lines += 1
                                matches.append(preview)
                    except Exception:
                        continue
            if not matches:
                related = suggest_related_files(root, pattern, glob_patterns)
                hint = "No exact text match. Try a broader pattern, inspect related files, or search for nearby runtime symbols."
                if related:
                    hint += f" Related files: {', '.join(related[:5])}"
                return {"ok": True, "matches": [], "hint": hint, "related_files": related[:8]}
            if len(matches) > MAX_MATCH_PREVIEW:
                preview = matches[:MAX_MATCH_PREVIEW]
                spill_path = write_spill(
                    "Grep",
                    {
                        "tool": "Grep",
                        "workspace": str(workspace),
                        "path": str(args.get("path", ".")),
                        "glob": str(args.get("glob", "*")),
                        "pattern": pattern,
                        "total_matches": len(full_matches),
                        "matches": full_matches,
                    },
                )
                response: dict[str, Any] = {
                    "ok": True,
                    "matches": preview,
                    "truncated": True,
                    "total_matches": len(matches),
                    "omitted_matches": len(matches) - len(preview),
                    "output_path": spill_path,
                    "hint": "Result list is long. Showing preview only; full matches were spilled to output_path.",
                    "read_suggestions": build_read_suggestions(preview),
                }
                if trimmed_lines > 0:
                    response["trimmed_line_texts"] = trimmed_lines
                return response
            response = {
                "ok": True,
                "matches": matches,
                "hint": "Use Read on a small window around one of the matched lines instead of reading from offset=1.",
                "read_suggestions": build_read_suggestions(matches),
            }
            if trimmed_lines > 0:
                response["trimmed_line_texts"] = trimmed_lines
            return response

        if tool == "Write":
            file_path = resolve_inside_workspace(str(args["file_path"]))
            content = str(args["content"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"ok": True, "path": str(file_path), "bytes": len(content.encode("utf-8"))}

        if tool == "Edit":
            file_path = resolve_inside_workspace(str(args["file_path"]))
            old_string = str(args["old_string"])
            new_string = str(args["new_string"])
            content = file_path.read_text(encoding="utf-8")
            if old_string not in content:
                return {"ok": False, "error": "old_string not found"}
            file_path.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
            return {"ok": True, "path": str(file_path)}

        if tool == "Bash":
            command = str(args["command"]).strip()
            command = _translate_bash_command(command)
            completed = subprocess.run(command, cwd=workspace, shell=True, capture_output=True, text=True, timeout=180, check=False)
            output = completed.stdout.strip()
            if completed.stderr.strip():
                output = f"{output}\n{completed.stderr.strip()}".strip()
            return {"ok": completed.returncode == 0, "output": output, "returncode": completed.returncode}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": f"Unknown tool: {tool}"}
