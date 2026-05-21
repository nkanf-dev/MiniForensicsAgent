# MiniForensicsAgent

## Dev setup
```bash
uv sync
uv run pytest          # all tests
uv run pytest tests/test_tools.py  # single test file
```

## Run
```bash
uv run mini-forensics-agent --model LocoOperator --workspace /path/to/workspace --task "..."
uv run mini-forensics-agent-tui
./scripts/run-mini-forensics-agent --help
```

## Architecture
- Package: `src/miniforensicsagent`
- Scripts entry: `mini-forensics-agent = "miniforensicsagent.cli:main"`
- Agent loop: `src/miniforensicsagent/loop.py`
- Tools (Read, Glob, Grep, Bash, Write, Edit, ActivateSkill, ReadSkillResource): `src/miniforensicsagent/tools.py`
- Skills discovery: `src/miniforensicsagent/skills.py`
- Model loading: `src/miniforensicsagent/models.py`

## Model discovery
- Default root: `~/.lmstudio/models`
- Use `--list-models` to see discovered models without running a task
- Pass a model name, index, or absolute path

### llama.cpp backend
```bash
# List available models from llama.cpp server
uv run mini-forensics-agent --engine llamacpp --list-models

# Run with llama.cpp server (plain completion API)
uv run mini-forensics-agent --engine llamacpp --llama-cpp-url http://localhost:8080/v1 --llama-cpp-model YOUR_MODEL.gguf --task "..." --workspace .

# Run with chat API (Jinja/chat template models)
uv run mini-forensics-agent --engine llamacpp --use-chat --llama-cpp-url http://localhost:8080/v1 --llama-cpp-model YOUR_MODEL.gguf --task "..." --workspace . --stream
```
- Default URL: `http://localhost:8080/v1`
- Default model: `qwen3.5-9b-instruct.Q4_K_M_deepseek4.gguf`
- Plain mode: uses `/v1/completions` endpoint
- Chat mode (`--use-chat`): uses `/v1/chat/completions` with OpenAI-style messages
- `tokenizer = None` (no token counting, `--stream` recommended for live output)

### Windows compatibility
- Glob/Grep: uses Python fallback when ripgrep (`rg`) is not available
- Bash: translates common commands (`pwd`→`cd`, `ls`→`dir`, `cat`→`type`, etc.)

## Tools quirks
- `Read` offset is **1-indexed**
- `Read` limit can be `"end"` (meaning "rest of file")
- `Glob` and `Grep` use `rg` (ripgrep); `Glob` also calls `rg --files` — both have 180s timeout
- Truncated results spill to `.mini_forensics_spill/` under the workspace
- `Bash` runs with `shell=True` and cwd=workspace; restricted access

## Skills system
- Skill roots (precedence low→high): `~/.agents/skills`, `~/.mini-forensics-agent/skills`, `<workspace>/.agents/skills`, `<workspace>/.mini-forensics-agent/skills`
- Each skill dir must contain `SKILL.md` with YAML frontmatter (`name`, `description`)
- `--list-skills` shows discovered skills; requires `--enable-skills` or implies it
- Skills activated at runtime via `ActivateSkill` tool

## Experimental flags
- `--enable-skills` — enable skills discovery and activation tools
- `--multi-tool` — allow multiple independent tool calls per turn
- `--compress-observations` — compress old observation payloads (O(n²)→O(n) tokens)
- `--transcript-window K` — include only last K turns per prompt
- `--turboquant` — enable TurboQuant KV cache backend (requires `uv sync --extra turboquant`)
- `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` — KV cache quantization via effGen