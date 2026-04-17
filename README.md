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

Experimental Agent Skills scaffold:

```bash
uv run mini-forensics-agent \
  --model 'LocoOperator-4B-mlx-4Bit' \
  --workspace /path/to/workspace \
  --task 'Inspect this repo and use any matching skill if needed.' \
  --enable-skills \
  --stream
```

Skill discovery roots, in increasing precedence:
- `~/.agents/skills`
- `~/.mini-forensics-agent/skills`
- `<workspace>/.agents/skills`
- `<workspace>/.mini-forensics-agent/skills`
- any extra `--skill-dir /path/to/skills`

Each skill lives in its own directory and must include a `SKILL.md` with frontmatter:

```md
---
name: demo-skill
description: One-line summary of when the skill should be used
---

# Instructions
```

Useful skill commands:
- `uv run mini-forensics-agent --workspace /path/to/workspace --list-skills`
- `uv run mini-forensics-agent --workspace /path/to/workspace --enable-skills --skill-dir /extra/skills ...`

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
