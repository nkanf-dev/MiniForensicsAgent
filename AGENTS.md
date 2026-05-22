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
- Agent loop: `loop.py`
- Tools: `tools.py` (Read, Glob, Grep, Bash, Write, Edit, ActivateSkill, ReadSkillResource)
- Skills: `skills.py`
- Model loading: `models.py`, `llamacpp.py`

## GUI / TUI Parameters
- Engine selector: dropdown/radio (MLX / llama.cpp)
- llama.cpp URL, model, use_chat always visible
- When engine=llamacpp: **use_chat is forced True** in both GUI and TUI
- GUI conversation list: click × to delete single conversation

## llama.cpp backend
- Default: `--llama-cpp-url http://localhost:8080/v1`
- Chat mode (`--use-chat`): `/v1/chat/completions` with jinja template → structured `<final>` output
- Plain mode (`--use-chat` omitted): `/v1/completions` → pure text continuation, **cannot produce structured output**
- **llama.cpp use_chat must be True** for agent loop to parse `<final>` / `<tool_call>` tags
- `--stream` recommended for live token output

## Windows compatibility
- Glob/Grep: Python fallback when `rg` (ripgrep) unavailable
- Bash: auto-translates `pwd`→`cd`, `ls`→`dir`, `cat`→`type`, `find`→`dir /s`

## Tool quirks
- `Read` offset is **1-indexed**, limit is `"end"` for rest of file
- Tool calls: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
- Observations: `tool` role, truncated at **2000 chars**
- Truncated results spill to `.mini_forensics_spill/`
- `Bash` restricted to allowed commands; translated for Windows

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
- GUI: click × on conversation card to delete