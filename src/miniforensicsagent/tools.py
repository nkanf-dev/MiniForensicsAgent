from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


ALLOWED_BASH = {"pwd", "ls", "find", "cat", "grep", "head", "xargs"}
DEFAULT_READ_LIMIT = 80
FORBIDDEN_BASH_SNIPPETS = {"rm ", "mv ", "cp ", "chmod ", "chown ", "curl ", "wget ", "python ", "python3 ", "node ", "npm ", "sh ", "bash ", ">>", " >", "< "}


def expand_brace_pattern(pattern: str) -> list[str]:
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    expanded: list[str] = []
    for option in match.group(1).split(","):
        expanded.extend(expand_brace_pattern(pattern[: match.start()] + option + pattern[match.end() :]))
    return expanded


def run_tool(call: dict[str, Any], workspace: Path) -> dict[str, Any]:
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

    try:
        if tool == "Read":
            file_path = resolve_inside_workspace(str(args["file_path"]))
            offset = max(1, int(args.get("offset", args.get("start_line", 1))))
            limit = max(1, int(args.get("limit", args.get("max_lines", DEFAULT_READ_LIMIT))))
            lines = file_path.read_text(encoding="utf-8").splitlines()
            start_index = offset - 1
            end_index = start_index + limit
            chunk = lines[start_index:end_index]
            return {"ok": True, "content": "\n".join(chunk), "offset": offset, "limit": limit, "returned_lines": len(chunk), "total_lines": len(lines), "truncated": end_index < len(lines)}

        if tool == "Glob":
            root = resolve_inside_workspace(str(args.get("path", ".")))
            if not root.exists():
                return {"ok": False, "error": f"Path does not exist: {root}", "cwd_hint": str((root.parent if root.parent.exists() else workspace).resolve()), "entries": nearby_listing(root)}
            patterns = [item.replace(".*", "*") for item in expand_brace_pattern(str(args["pattern"]))]
            matches: list[str] = []
            for pattern in patterns:
                completed = subprocess.run(["rg", "--files", str(root), "-g", pattern], cwd=workspace, capture_output=True, text=True, timeout=10, check=False)
                for line in completed.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        matches.append(str(Path(line).resolve().relative_to(workspace)))
                    except Exception:
                        continue
            matches = sorted(dict.fromkeys(matches))[:200]
            if not matches:
                related = suggest_related_paths_for_glob(root, str(args["pattern"]))
                hint = "No glob matches. Try a broader pattern, inspect nearby directories, or search a parent path first."
                if related:
                    hint += f" Related paths: {', '.join(related[:5])}"
                return {"ok": True, "matches": [], "hint": hint, "related_files": related[:8]}
            return {"ok": True, "matches": matches}

        if tool == "Grep":
            root = resolve_inside_workspace(str(args.get("path", ".")))
            if not root.exists():
                return {"ok": False, "error": f"Path does not exist: {root}", "cwd_hint": str((root.parent if root.parent.exists() else workspace).resolve()), "entries": nearby_listing(root)}
            pattern = str(args["pattern"])
            glob_patterns = expand_brace_pattern(str(args.get("glob", "*")))
            rg_cmd = ["rg", "-n", "-S", "--no-heading", pattern]
            for glob_pattern in glob_patterns:
                rg_cmd.extend(["-g", glob_pattern])
            rg_cmd.append(str(root))
            completed = subprocess.run(rg_cmd, cwd=workspace, capture_output=True, text=True, timeout=10, check=False)
            matches: list[dict[str, Any]] = []
            for line in completed.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                file_part, line_no, text = parts
                try:
                    rel = str(Path(file_part).resolve().relative_to(workspace))
                except Exception:
                    rel = file_part
                matches.append({"file": rel, "line": int(line_no) if line_no.isdigit() else 1, "text": text})
            if not matches:
                related = suggest_related_files(root, pattern, glob_patterns)
                hint = "No exact text match. Try a broader pattern, inspect related files, or search for nearby runtime symbols."
                if related:
                    hint += f" Related files: {', '.join(related[:5])}"
                return {"ok": True, "matches": [], "hint": hint, "related_files": related[:8]}
            trimmed = matches[:200]
            return {"ok": True, "matches": trimmed, "hint": "Use Read on a small window around one of the matched lines instead of reading from offset=1.", "read_suggestions": build_read_suggestions(trimmed)}

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
            parts = shlex.split(command)
            if not parts or parts[0] not in ALLOWED_BASH:
                return {"ok": False, "error": f"Command not allowed: {command}"}
            lowered = f" {command.lower()} "
            if any(snippet in lowered for snippet in FORBIDDEN_BASH_SNIPPETS):
                return {"ok": False, "error": f"Command not allowed: {command}"}
            completed = subprocess.run(command, cwd=workspace, shell=True, capture_output=True, text=True, timeout=10, check=False)
            output = completed.stdout.strip()
            if completed.stderr.strip():
                output = f"{output}\n{completed.stderr.strip()}".strip()
            return {"ok": completed.returncode == 0, "output": output, "returncode": completed.returncode}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": f"Unknown tool: {tool}"}
