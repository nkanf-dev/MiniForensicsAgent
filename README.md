# MiniForensicsAgent

A small local forensics-style explorer loop for MLX models.

## What it does

- Runs a local model in a tool-using loop
- Searches a workspace with `Glob`, `Grep`, `Read`, and a restricted `Bash`
- Prefers runtime evidence over weak clues
- Uses Claude Code-style `Read(file_path, offset, limit)`
- Shows streaming progress and observations with `rich`
- Includes a Textual TUI with parameter popup, run history, and observability

## Install

```bash
cd MiniForensicsAgent
uv sync
```

Optional TurboQuant backend:

```bash
uv sync --extra turboquant
```

## Run

```bash
uv run mini-forensics-agent \
  --model 'LocoOperator-4B-mlx-4Bit' \
  --workspace /path/to/workspace \
  --task 'Find the actual config used by this app.' \
  --stream
```

Or use the helper script:

```bash
./scripts/run-mini-forensics-agent --help
```

## TUI

```bash
uv run mini-forensics-agent-tui
```

TUI keys:
- `r` run task
- `s` stop current run
- `n` new conversation
- `p` open parameter popup
- `q` quit
