# MiniForensicsAgent

## Dev setup
```bash
uv sync
uv run pytest                    # all tests (25 passing, 1 skipped)
uv run pytest tests/test_prompting.py  # single test file
```

## Run
```bash
# MLX (Apple Silicon)
uv run mini-forensics-agent --model LocoOperator --workspace . --task "..."

# llama.cpp (any platform)
uv run mini-forensics-agent --engine llamacpp --use-chat --llama-cpp-url http://localhost:8080/v1 --llama-cpp-model MODEL.gguf --task "..." --workspace . --stream

# TUI / GUI
uv run mini-forensics-agent-tui
uv run mini-forensics-agent-gui
```

## Architecture
- Package: `src/miniforensicsagent`
- Entry: `mini-forensics-agent = "miniforensicsagent.cli:main"`
- Agent loop: `loop.py` (run_loop function)
- Tools: `tools.py` (Read, Glob, Grep, Bash, Write, Edit, ActivateSkill, ReadSkillResource)
- Skills: `skills.py`
- Model loading: `models.py`, `llamacpp.py`

## Loop exit conditions
Loop exits via `LoopResult` in these cases:
1. `response["type"] == "final"` → `LoopResult(True, answer, ...)` ✅ success
2. `should_stop()` returns True → `LoopResult(False, "cancelled", ...)`
3. `max_iterations` reached → `LoopResult(False, "max_iterations_reached", ...)`

The `normalize_response()` function (loop.py:164) determines response type:
- `payload.get("type") == "final"` or `payload.get("action") == "final"` → type="final", finish="explicit"
- rubric with `finish_when`/`strong_evidence`/`weak_evidence` → normalized to plan type
- plan/plan_update payloads normalized via `parse_plan_payload()`
- `finish_reason="stop"` with no tool_calls → type="final", finish="implicit" (new!)
- Blocking tool failures before final cause retry (loop.py:624)

## Doom loop detection
- Trigger: same tool + same args called 3 times within 60 seconds
- Warning event: `{"type": "doom_loop_warning", "tool": "...", "args": {...}, "choices": ["once", "always", "reject"]}`
- Whitelist: `.mini_forensics_doom_whitelist.json` in working directory (5 min TTL, persisted)
- Timeout: 30 seconds (auto-chooses "once")
- `choice="always"` adds to whitelist via `_add_to_whitelist()` (loop.py:822-840)

### Doom Loop UI Dialogs
- **TUI**: `DoomLoopModal` — Textual ModalScreen with Once/Always/Reject buttons
- **GUI**: `DoomLoopDialog` — NiceGUI ui.dialog with Once/Always/Reject buttons
- User clicks button → `trigger_doom_loop_choice(choice)` notifies DoomLoopWaiter
- Callback mechanism: `set_doom_loop_global_callback()` + `trigger_doom_loop_choice()`

## GUI / TUI Parameters
- Engine selector: Select dropdown (TUI), radio buttons (GUI) — MLX / llama.cpp
- llama.cpp URL, model always visible
- When engine=llamacpp: **use_chat is forced True** in both GUI and TUI
- GUI: click × on conversation card to delete

## llama.cpp backend
- Default: `--llama-cpp-url http://localhost:8080/v1`
- Chat mode (`--use-chat`): `/v1/chat/completions` → structured `<final>` output
- Plain mode (no `--use-chat`): `/v1/completions` → plain text, **cannot produce structured output**
- **use_chat must be True** for agent loop to parse `<final>` / `<tool_call>` tags
- `--stream` recommended for live token output

## Windows compatibility
- Glob/Grep: Python fallback when `rg` (ripgrep) unavailable (tools.py:20)
- Bash: auto-translates `pwd`→`cd`, `ls`→`dir`, `ls -la`→`dir /a`, `cat`→`type`, `find`→`dir /s /b`
- Full command match first, then base token match (tools.py:42-51)

## Tool quirks
- `Read` offset is **1-indexed**, limit is `"end"` for rest of file
- Tool call format: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
- Observations truncated at **2000 chars**, spilled to `.mini_forensics_spill/`
- `Bash` restricted to allowed commands (translated for Windows)

## Prompt format
- First turn must return plan JSON: `{"type":"plan","goal":"...","steps":[...],"done_when":"..."}`
- Final answer: `<final>{"answer":"..."}</final>` — do NOT output plain text before final
- Tool results in chat mode: full JSON serialized (prompting.py:build_chat_messages)

## Skills system
- Skill roots: `~/.agents/skills`, `~/.mini-forensics-agent/skills`, `<workspace>/.agents/skills`
- Each skill dir: `SKILL.md` with YAML frontmatter (`name`, `description`)
- Activate with `ActivateSkill(skill_name)` tool

## Experimental flags
- `--enable-skills` — enable skills
- `--multi-tool` — allow multiple tool calls per turn
- `--compress-observations` — compress transcript
- `--transcript-window K` — limit transcript to last K turns
- `--turboquant` — KV cache quantization (requires `uv sync --extra turboquant`)

## Conversation persistence
- Stored in `.mini_forensics_conversations.json` (in working directory)
- Auto-saved on every run completion
- Auto-loaded on GUI/TUI startup

## Technical Debt

### Resolved (commit 49bcdc8)
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

### Outstanding

#### P1 - High Priority
| ID | 问题 | 文件 | 建议 |
|----|------|------|------|
| TD-01 | run_tool if/elif chain (~150 lines) | tools.py | Dict dispatch pattern |
| TD-02 | Missing type hints (Any泛滥) | Multiple | Add incrementally |
| TD-03 | Inconsistent error handling | Multiple | Unify Result type |

#### P2 - Medium Priority
| ID | 问题 | 文件 | 建议 |
|----|------|------|------|
| TD-04 | Config classes scattered | gui.py, cli.py | Extract Config abstraction |
| TD-05 | Skills tight coupling | skills.py | Separate responsibilities |
| TD-06 | Model loading engine coupling | models.py | Strategy pattern |

#### P3 - Low Priority
| ID | 问题 | 文件 | 建议 |
|----|------|------|------|
| TD-07 | count_tokens re-joins chunks | loop.py | Incremental counting |
| TD-08 | Bash AllowedCommands scattered | prompting.py | Centralize constant |
| TD-09 | _cleanup_expired_entries modifies during iteration | loop.py | Collect keys first |