# Technical Debt - MiniForensicsAgent

## Resolved (commit 49bcdc8)
| ID | 问题 | 修复 |
|----|------|------|
| P0-1 | GUI SessionState per-request instantiation | Module-level singleton |
| P0-2 | TUI last conversation delete warning | Already had return |
| P0-3 | Windows path separator `\`, `/` mixed | `rel.as_posix()` |
| P0-4 | Pipe operator `\|` not translated on Windows | Recursive translation |
| P1-5/6 | Symlink path traversal | Resolve before comparison |
| P1-7 | Windows drive letter `C:` handling | Added detection |
| P1-9 | ValueError raised in extract_first_json_object | Return None instead |
| P1-10 | discover_models rglob(*) performance | rglob(config.json) |
| P1-11 | Whitelist race conditions | threading.Lock added |

---

## Outstanding

### P1 - High Priority
| ID | 问题 | 文件 | 复杂度 | 建议 |
|----|------|------|--------|------|
| TD-01 | run_tool if/elif chain (~150 lines) | tools.py | 中 | Dict dispatch pattern |
| TD-02 | Missing type hints (Any泛滥) | Multiple | 高 | Add incrementally |
| TD-03 | Inconsistent error handling | Multiple | 中 | Unify Result type |

### P2 - Medium Priority
| ID | 问题 | 文件 | 复杂度 | 建议 |
|----|------|------|--------|------|
| TD-04 | Config classes scattered | gui.py, cli.py | 中 | Extract Config abstraction |
| TD-05 | Skills tight coupling | skills.py | 中 | Separate responsibilities |
| TD-06 | Model loading engine coupling | models.py | 中 | Strategy pattern |

### P3 - Low Priority
| ID | 问题 | 文件 | 复杂度 | 建议 |
|----|------|------|--------|------|
| TD-07 | count_tokens re-joins chunks every 20 | loop.py | 低 | Incremental counting |
| TD-08 | Bash AllowedCommands scattered | prompting.py | 低 | Centralize constant |
| TD-09 | _cleanup_expired_entries modifies during iteration | loop.py | 低 | Collect keys first |

---

## Details

### TD-01: run_tool if/elif chain
**Location**: `tools.py:71-400`
**Issue**: ~150 lines of if/elif/elif/... for each tool type
**Fix**: Replace with dict dispatch:
```python
_TOOL_HANDLERS = {
    "Read": _handle_read,
    "Glob": _handle_glob,
    "Grep": _handle_grep,
    "Write": _handle_write,
    "Edit": _handle_edit,
    "Bash": _handle_bash,
    "ActivateSkill": _handle_activate_skill,
    "ReadSkillResource": _handle_read_skill_resource,
}

def run_tool(call, ...):
    tool = call["name"]
    handler = _TOOL_HANDLERS.get(tool)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {tool}"}
    return handler(call, workspace, ...)
```

### TD-02: Missing type hints
**Location**: Multiple files
**Issue**: Heavy use of `Any` weakens type safety, hinders refactoring
**Fix**: Add types incrementally, prefer concrete types over Any

### TD-03: Inconsistent error handling
**Location**: Multiple files
**Issue**: Some functions return `None`, some return `{"ok": False, "error": "..."}`, some raise exceptions
**Fix**: Consider Result pattern:
```python
from result import Result, Ok, Err

def some_function() -> Result[GoodType, str]:
    if success:
        return Ok(value)
    return Err("error message")
```

### TD-04: Config classes scattered
**Location**: `gui.py`, `cli.py`
**Issue**: `GuiConfig`, CLI args processed separately
**Fix**: Extract unified `AgentConfig` class

### TD-05: Skills tight coupling
**Location**: `skills.py`
**Issue**: `SkillCatalog`, `Skill`, filesystem operations all in one class
**Fix**: Separate into `SkillDiscovery`, `SkillLoader`, `Skill` entities

### TD-06: Model loading engine coupling
**Location**: `models.py`
**Issue**: if/else for MLX vs llama.cpp in `load_local_model`
**Fix**: Strategy pattern with `ModelEngine` base class

### TD-07: Token count re-computation
**Location**: `loop.py:557, 603`
**Issue**: `"".join(chunks)` creates new string every 20 tokens
**Fix**: Maintain running token count

### TD-08: AllowedCommands scattered
**Location**: `prompting.py`, `tools.py`
**Issue**: Command whitelist not centralized
**Fix**: Single constant `ALLOWED_COMMANDS` in constants module

### TD-09: Dict modification during iteration
**Location**: `loop.py:134-136`
**Issue**: `del _doom_loop_whitelist[k]` while iterating
**Status**: Fixed in 49bcdc8 (collect keys first)