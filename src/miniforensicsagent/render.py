from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
    RICH_STDERR = Console(stderr=True)
except Exception:
    HAS_RICH = False
    Console = Group = Live = Panel = Table = Text = None  # type: ignore[assignment]
    RICH_STDERR = None


SPINNER_FRAMES = ["[=     ]", "[==    ]", "[===   ]", "[ ===  ]", "[  === ]", "[   ===]", "[    ==]", "[     =]"]


def count_tokens(tokenizer: Any, text: str) -> int | None:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return None


def start_prefill_indicator(iteration: int, prompt_tokens: int | None) -> tuple[threading.Event, threading.Thread, float]:
    started = time.perf_counter()
    stop_event = threading.Event()

    def worker() -> None:
        index = 0
        while not stop_event.is_set():
            elapsed = time.perf_counter() - started
            token_text = f"{prompt_tokens}" if prompt_tokens is not None else "?"
            frame = SPINNER_FRAMES[index % len(SPINNER_FRAMES)]
            print(
                f"\r[iteration {iteration}] prefill {frame} prompt_tokens={token_text} waiting_for_first_token {elapsed:5.1f}s",
                end="",
                file=sys.stderr,
                flush=True,
            )
            index += 1
            stop_event.wait(0.12)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event, thread, started


def build_status_renderable(iteration: int, *, phase: str, prompt_tokens: int | None, first_token_latency: float | None, generated_tokens: int, elapsed: float, tps: float, raw_preview: str):
    if not HAS_RICH:
        return None
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="cyan", justify="right")
    stats.add_column(style="white")
    stats.add_row("Iteration", str(iteration))
    stats.add_row("Phase", phase)
    stats.add_row("Prompt Tokens", str(prompt_tokens) if prompt_tokens is not None else "?")
    stats.add_row("First Token", f"{first_token_latency:.2f}s" if first_token_latency is not None else "waiting")
    stats.add_row("Generated", str(generated_tokens))
    stats.add_row("Elapsed", f"{elapsed:.2f}s")
    stats.add_row("Tok/s", f"{tps:.1f}")
    preview_text = Text(raw_preview[-500:] if raw_preview else "(no output yet)", overflow="fold")
    return Group(Panel(stats, title=f"Iteration {iteration}", border_style="cyan"), Panel(preview_text, title="Model Output", border_style="magenta"))


def emit_observation_rendered(iteration: int, decision: dict[str, Any], observation: dict[str, Any]) -> bool:
    if not HAS_RICH or RICH_STDERR is None:
        return False
    table = Table.grid(padding=(0, 1))
    table.add_column(style="cyan", justify="right")
    table.add_column(style="white")
    if decision:
        table.add_row("Decision", str(decision.get("name", decision.get("type", "unknown"))))
    for key in ("error", "hint", "cwd_hint"):
        if observation.get(key):
            table.add_row(key, str(observation.get(key)))
    if "matches" in observation and isinstance(observation.get("matches"), list):
        matches = observation["matches"]
        table.add_row("Matches", json.dumps(matches[:5], ensure_ascii=False))
        if len(matches) > 5:
            table.add_row("More", f"+{len(matches) - 5} more")
    if observation.get("related_files"):
        table.add_row("Related", ", ".join(str(item) for item in observation["related_files"][:6]))
    if observation.get("switch_file_suggestions"):
        table.add_row("Switch File", ", ".join(str(item) for item in observation["switch_file_suggestions"][:6]))
    if observation.get("read_suggestions"):
        table.add_row("Next Read", json.dumps(observation["read_suggestions"][:3], ensure_ascii=False))
    if observation.get("suggested_read"):
        table.add_row("Suggested Read", json.dumps(observation["suggested_read"], ensure_ascii=False))
    if "offset" in observation:
        table.add_row("Read Window", f"offset={observation.get('offset')} limit={observation.get('limit')} returned={observation.get('returned_lines')} total={observation.get('total_lines')}")
    if "content" in observation:
        table.add_row("Content", str(observation.get("content", ""))[:500])
    if "output" in observation:
        table.add_row("Output", str(observation.get("output", ""))[:500])
    RICH_STDERR.print(Panel(table, title=f"Iteration {iteration} Observation", border_style="green" if observation.get("ok") else "red"))
    return True
