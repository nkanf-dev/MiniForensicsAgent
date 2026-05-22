# MiniForensicsAgent

## Dev setup
```bash
uv sync
uv run pytest                    # all tests (24 passing)
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

## llama.cpp backend
- Default: `--llama-cpp-url http://localhost:8080/v1`
- Plain mode (`--engine llamacpp` without `--use-chat`): `/v1/completions`
- Chat mode (`--use-chat`): `/v1/chat/completions` with OpenAI-style messages
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