from __future__ import annotations

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .evidence import (
    has_promising_but_incomplete_candidate,
    update_evidence_cache,
)
from .prompting import build_chat_messages, build_prompt
from .render import HAS_RICH, Live, RICH_STDERR, build_status_renderable, count_tokens, emit_observation_rendered, start_prefill_indicator
from .skills import SkillCatalog, render_active_skill_context, render_skill_catalog
from .tools import run_tool


TOOL_NAMES = {"Read", "Glob", "Grep", "Bash", "Write", "Edit", "ActivateSkill", "ReadSkillResource"}

DOOM_LOOP_THRESHOLD = 3
DOOM_LOOP_WINDOW_SECONDS = 60
DOOM_LOOP_WHITELIST_DURATION_SECONDS = 300
DOOM_LOOP_RESPONSE_TIMEOUT_SECONDS = 30
DOOM_LOOP_WHITELIST_FILE = ".mini_forensics_doom_whitelist.json"


@dataclass
class ToolCallRecord:
    name: str
    args_frozen: frozenset
    timestamp: float


_doom_loop_whitelist: dict[tuple[str, frozenset], float] = {}
_doom_loop_waiter: "DoomLoopWaiter | None" = None


class DoomLoopWaiter:
    def __init__(self):
        self._event: threading.Event | None = None
        self._choice: str | None = None

    def wait(self, timeout: int = DOOM_LOOP_RESPONSE_TIMEOUT_SECONDS) -> str:
        if self._event is None:
            return "once"
        if self._event.wait(timeout):
            return self._choice or "once"
        return "once"

    def notify(self, choice: str) -> None:
        self._choice = choice
        if self._event:
            self._event.set()


def _build_args_frozen(args: dict) -> frozenset:
    if not isinstance(args, dict):
        return frozenset()
    return frozenset((k, str(v)) for k, v in sorted(args.items()))


def _load_whitelist() -> None:
    global _doom_loop_whitelist
    path = Path(DOOM_LOOP_WHITELIST_FILE)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            now = time.time()
            _doom_loop_whitelist = {
                (k[0], frozenset(k[1])): v
                for k, v in data.items()
                if now - v < DOOM_LOOP_WHITELIST_DURATION_SECONDS
            }
        except (json.JSONDecodeError, ValueError):
            _doom_loop_whitelist = {}
    else:
        _doom_loop_whitelist = {}


def _save_whitelist() -> None:
    path = Path(DOOM_LOOP_WHITELIST_FILE)
    data = {list(k): v for k, v in _doom_loop_whitelist.items()}
    path.write_text(json.dumps(data))


def _is_whitelisted(name: str, args: dict) -> bool:
    key = (name, _build_args_frozen(args))
    if key not in _doom_loop_whitelist:
        return False
    if time.time() - _doom_loop_whitelist[key] > DOOM_LOOP_WHITELIST_DURATION_SECONDS:
        del _doom_loop_whitelist[key]
        return False
    return True


def _add_to_whitelist(name: str, args: dict) -> None:
    key = (name, _build_args_frozen(args))
    _doom_loop_whitelist[key] = time.time()
    _save_whitelist()


def _check_doom_loop(name: str, args: dict, recent_calls: list[ToolCallRecord]) -> bool:
    if _is_whitelisted(name, args):
        return False
    if len(recent_calls) < DOOM_LOOP_THRESHOLD:
        return False
    args_key = _build_args_frozen(args)
    recent_same = [
        r for r in recent_calls[-DOOM_LOOP_THRESHOLD:]
        if r.name == name and r.args_frozen == args_key
    ]
    return len(recent_same) >= DOOM_LOOP_THRESHOLD


def _cleanup_expired_entries(recent_calls: list[ToolCallRecord]) -> list[ToolCallRecord]:
    now = time.time()
    cutoff = now - DOOM_LOOP_WINDOW_SECONDS
    expired = [k for k, v in _doom_loop_whitelist.items() if now - v > DOOM_LOOP_WHITELIST_DURATION_SECONDS]
    for k in expired:
        del _doom_loop_whitelist[k]
    if expired:
        _save_whitelist()
    return [r for r in recent_calls if r.timestamp > cutoff]


def _wait_doom_loop_response() -> str:
    global _doom_loop_waiter
    if _doom_loop_waiter is None:
        return "once"
    choice = _doom_loop_waiter.wait(DOOM_LOOP_RESPONSE_TIMEOUT_SECONDS)
    _doom_loop_waiter = None
    return choice


def _trigger_doom_loop_wait(tool: str, args: dict, iteration: int, event_callback: Callable | None) -> str:
    global _doom_loop_waiter
    _doom_loop_waiter = DoomLoopWaiter()
    _doom_loop_waiter._event = threading.Event()
    if event_callback:
        event_callback({
            "type": "doom_loop_warning",
            "tool": tool,
            "args": args,
            "iteration": iteration,
            "choices": ["once", "always", "reject"],
            "timeout": DOOM_LOOP_RESPONSE_TIMEOUT_SECONDS,
        })
    return _wait_doom_loop_response()


@dataclass
class LoopResult:
    success: bool
    answer: str
    iterations: int
    tool_calls: int
    transcript: list[dict[str, Any]]


def default_plan_state() -> dict[str, Any]:
    return {
        "goal": "find one reliable answer",
        "steps": ["discover candidate files", "verify runtime usage", "finish with evidence-backed conclusion"],
        "done_when": "one answer is supported by trusted evidence",
        "completed_steps": [],
        "current_step": "discover candidate files",
    }


def parse_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fallback = default_plan_state()
    goal = str(payload.get("goal", "")).strip() or fallback["goal"]
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        steps = list(fallback["steps"])
    else:
        steps = [str(step).strip() for step in raw_steps if str(step).strip()][:8]
        if not steps:
            steps = list(fallback["steps"])
    done_when = str(payload.get("done_when", "")).strip() or fallback["done_when"]
    completed_raw = payload.get("completed_steps")
    completed_steps = [str(step).strip() for step in completed_raw] if isinstance(completed_raw, list) else []
    current_step = str(payload.get("current_step", "")).strip() or (steps[0] if steps else "")
    normalized_completed = [step for step in completed_steps if step in steps]
    return {
        "goal": goal,
        "steps": steps,
        "done_when": done_when,
        "completed_steps": normalized_completed,
        "current_step": current_step,
    }


def summarize_evidence(cache: dict[str, dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for value, payload in cache.items():
        if value == "__symbols__" or payload.get("artifact_type") != "candidate":
            continue
        evidence = payload.get("evidence", [])
        trusted_usage = sum(1 for item in evidence if item.get("trusted") and item.get("role") in {"usage", "success"})
        trusted_value = sum(1 for item in evidence if item.get("trusted") and item.get("role") == "value")
        score = trusted_usage * 3 + trusted_value
        if score <= 0:
            continue
        summary.append(
            {
                "value": value,
                "score": score,
                "trusted_usage_hits": trusted_usage,
                "trusted_value_hits": trusted_value,
            }
        )
    summary.sort(key=lambda item: item["score"], reverse=True)
    return summary[:limit]


def compute_tool_stats(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failures = 0
    for turn in transcript:
        decision = turn.get("decision", {})
        if decision.get("type") == "tool":
            name = str(decision.get("name", "tool"))
            counts[name] = counts.get(name, 0) + 1
        for observation in turn.get("observations", []):
            if observation.get("ok") is False:
                failures += 1
    return {"counts": counts, "failures": failures}


def _update_tool_stats(stats: dict[str, Any], turn: dict[str, Any]) -> None:
    """Incrementally update running tool stats with one turn."""
    decision = turn.get("decision", {})
    # Support both single decision and multi_tool decisions list.
    decisions = turn.get("decisions", [decision] if decision else [])
    for dec in decisions:
        if dec.get("type") == "tool":
            name = str(dec.get("name", "tool"))
            stats["counts"][name] = stats["counts"].get(name, 0) + 1
    for obs in turn.get("observations", []):
        if obs.get("ok") is False:
            stats["failures"] += 1


def extract_tagged_json(text: str, tag: str) -> dict[str, Any] | None:
    match = re.search(rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_first_json_object(text: str) -> dict[str, Any]:
    tagged_tool = extract_tagged_json(text, "tool_call")
    if tagged_tool:
        return tagged_tool
    tagged_final = extract_tagged_json(text, "final")
    if tagged_final:
        tagged_final.setdefault("type", "final")
        return tagged_final
    cleaned = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE))
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"No JSON object found in model output: {text[:300]}")


def extract_all_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all <tool_call> blocks from text (for multi_tool mode)."""
    calls = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                calls.append(parsed)
        except json.JSONDecodeError:
            pass
    return calls


def _extract_text_or_answer(payload: dict[str, Any]) -> str:
    if "answer" in payload:
        return str(payload["answer"])
    text = payload.get("text")
    if text and str(text).strip():
        return str(text)
    content = payload.get("content")
    if content and str(content).strip():
        return str(content)
    parts = payload.get("parts", [])
    if parts:
        texts = []
        for p in parts:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    texts.append(p["text"])
                elif p.get("type") == "content" and isinstance(p.get("content"), list):
                    for block in p["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
        if texts:
            return " ".join(t for t in texts if t)
    return "done"


def normalize_response(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("type") == "plan" or {"goal", "steps", "done_when"} <= set(payload.keys()):
        return {"type": "plan", "plan": parse_plan_payload(payload)}
    if payload.get("type") == "plan_update":
        return {"type": "plan_update", "plan": parse_plan_payload(payload)}
    if payload.get("type") == "rubric" or {"strong_evidence", "weak_evidence", "finish_when"} <= set(payload.keys()):
        goal = str(payload.get("finish_when", "finish with trusted evidence"))
        steps = [f"collect strong evidence: {item}" for item in payload.get("strong_evidence", [])] or default_plan_state()["steps"]
        mapped = {"goal": goal, "steps": steps, "done_when": goal}
        return {"type": "plan", "plan": parse_plan_payload(mapped)}
    if payload.get("type") == "final" or payload.get("action") == "final":
        return {"type": "final", "finish": "explicit", "answer": _extract_text_or_answer(payload)}
    finish = payload.get("finish_reason") or payload.get("finish")
    has_tool_calls = bool(payload.get("tool_calls"))
    if finish == "stop" and not has_tool_calls:
        return {"type": "final", "finish": "implicit", "answer": _extract_text_or_answer(payload)}
    name = payload.get("name") or payload.get("tool") or payload.get("action")
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else payload.get("args")
    if not isinstance(arguments, dict):
        arguments = {key: value for key, value in payload.items() if key not in {"name", "tool", "action", "arguments", "args", "type"}}
    if name in TOOL_NAMES:
        return {"type": "tool", "name": name, "arguments": arguments}
    raise ValueError(f"Unsupported tool/final payload: {payload}")


def is_empty_search_observation(decision: dict[str, Any], observation: dict[str, Any]) -> bool:
    if not observation.get("ok") or decision.get("type") != "tool":
        return False
    if decision.get("name") in {"Glob", "Grep"}:
        return observation.get("matches") == []
    if decision.get("name") == "Bash":
        return str(observation.get("output", "")).strip() == ""
    return False


def looks_like_narrow_search(decision: dict[str, Any]) -> bool:
    if decision.get("type") != "tool":
        return False
    tool = decision.get("name")
    args = decision.get("arguments", {})
    if tool in {"Glob", "Grep"}:
        pattern = str(args.get("pattern", ""))
        path = str(args.get("path", ""))
        wildcard_count = pattern.count("*") + pattern.count("{") + pattern.count("?")
        return wildcard_count <= 2 or path not in {"", "."}
    if tool == "Bash":
        command = str(args.get("command", ""))
        return "-name " in command or "grep " in command or "find " in command
    return False


def is_blocking_failure_for_final(observation: dict[str, Any]) -> bool:
    if observation.get("ok", True):
        return False
    return not observation.get("controller_guidance", False)


def compress_turn(turn: dict[str, Any]) -> None:
    """Replace large observation payloads with compact summaries in-place.

    Called after the turn is no longer the active context window, so the
    detailed content is no longer needed in subsequent prompts.
    """
    decision = turn.get("decision", {})
    if decision.get("type") != "tool":
        return
    tool_name = decision.get("name", "")
    iteration = turn.get("iteration", "?")
    args = decision.get("arguments", {})
    for obs in turn.get("observations", []):
        if not obs.get("ok"):
            continue
        if tool_name == "Read":
            content = obs.get("content", "")
            if content and len(content) > 200:
                file_path = args.get("file_path", "?")
                n_lines = obs.get("returned_lines", content.count("\n") + 1)
                obs["content"] = f"[compressed: Read {file_path} ~{n_lines} lines, seen at iter {iteration}]"
                obs.pop("truncated", None)
        elif tool_name in {"Grep", "Glob"}:
            matches = obs.get("matches")
            if isinstance(matches, list):
                pattern = args.get("pattern", "?")
                obs["matches"] = f"[compressed: {len(matches)} matches for {pattern!r}, seen at iter {iteration}]"
                obs.pop("read_suggestions", None)
                obs.pop("related_files", None)
        elif tool_name == "Bash":
            output = obs.get("output", "")
            if output and len(output) > 200:
                obs["output"] = f"[compressed: Bash output {len(output)} chars, seen at iter {iteration}]"


def update_active_skills(active_skills: dict[str, dict[str, Any]], decision: dict[str, Any], observation: dict[str, Any]) -> None:
    if decision.get("type") != "tool" or not observation.get("ok"):
        return
    if decision.get("name") == "ActivateSkill":
        skill_name = str(observation.get("name", "")).strip()
        if skill_name:
            active_skills[skill_name] = dict(observation)


def run_loop(
    model: Any,
    generation_config: Any,
    *,
    task: str,
    workspace: Path,
    max_iterations: int,
    max_tokens: int,
    temperature: float,
    stream_output: bool,
    reflection_strength: str,
    kv_bits: int | None = None,
    kv_group_size: int | None = None,
    quantized_kv_start: int | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    # Experimental features
    compress_observations: bool = False,
    transcript_window: int | None = None,
    multi_tool: bool = False,
    skill_catalog: SkillCatalog | None = None,
    use_chat: bool = False,
) -> LoopResult:
    transcript: list[dict[str, Any]] = []
    tool_calls = 0
    evidence_cache: dict[str, dict[str, Any]] = {}
    no_new_evidence_streak = 0
    narrow_empty_search_streak = 0
    plan_state: dict[str, Any] | None = None
    running_tool_stats: dict[str, Any] = {"counts": {}, "failures": 0}
    active_skills: dict[str, dict[str, Any]] = {}
    available_skills_block = render_skill_catalog(skill_catalog) if skill_catalog is not None else ""
    recent_tool_calls: list[ToolCallRecord] = []

    _load_whitelist()

    for iteration in range(1, max_iterations + 1):
        if should_stop is not None and should_stop():
            return LoopResult(False, "cancelled", max(1, iteration - 1), tool_calls, transcript)
        if event_callback is not None:
            event_callback({"type": "iteration_start", "iteration": iteration})
        reflection_hint = ""
        if reflection_strength != "none":
            previous_observations = transcript[-1].get("observations", []) if transcript else []
            latest_observation = previous_observations[-1] if previous_observations else {}
            latest_decision = transcript[-1].get("decision", {}) if transcript else {}
            hints: list[str] = []
            if len(transcript) >= 2 and transcript[-1].get("raw", "") == transcript[-2].get("raw", ""):
                hints.append("Your last two responses were identical. Change strategy now.")
            if no_new_evidence_streak >= 2:
                hints.append("Recent actions added no new evidence. Prefer a different source or finish if evidence is already sufficient.")
            if latest_observation and latest_observation.get("ok") is False:
                hints.append("The last action failed. Fix the failure instead of repeating the same command or path.")
            if latest_decision.get("type") == "tool" and latest_decision.get("name") == "Glob" and latest_observation.get("ok") and latest_observation.get("matches") == []:
                hints.append("The last glob was empty. Try Read/Grep on known candidate files or inspect a different directory.")
            if no_new_evidence_streak >= 3 and latest_decision.get("type") == "tool":
                hints.append("Repeated precise searches are not working. Back up one level and enumerate directories or use a broader pattern instead of guessing exact file names.")
            if reflection_strength in {"medium", "high"} and evidence_cache and no_new_evidence_streak >= 1:
                hints.append("Audit trusted evidence only: workflows, execution records, success logs. Ignore docs, backups, comments, and examples.")
            if reflection_strength in {"medium", "high"} and no_new_evidence_streak >= 2 and has_promising_but_incomplete_candidate(evidence_cache):
                hints.append("A promising candidate value exists, but usage proof is missing. Stop searching the same literal and follow nearby symbols or references to find where it is used.")
            if hints:
                reflection_hint = "\nReflection trigger:\n- " + "\n- ".join(hints) + "\n"

        tokenizer = getattr(model, "tokenizer", None)
        if use_chat:
            messages = build_chat_messages(
                task,
                transcript,
                workspace,
                current_plan=plan_state or default_plan_state(),
                remaining_iterations=max_iterations - iteration + 1,
                reflection_hint=reflection_hint,
                multi_tool=multi_tool,
                available_skills=available_skills_block,
                active_skill_context=render_active_skill_context(active_skills),
            )
            prompt_tokens = None
        else:
            prompt = build_prompt(
                task,
                transcript,
                workspace,
                current_plan=plan_state or default_plan_state(),
                remaining_iterations=max_iterations - iteration + 1,
                reflection_hint=reflection_hint,
                window=transcript_window,
                multi_tool=multi_tool,
                available_skills=available_skills_block,
                active_skill_context=render_active_skill_context(active_skills),
            )
            prompt_tokens = count_tokens(tokenizer, prompt) if tokenizer is not None else None
        config = generation_config(temperature=temperature, max_tokens=max_tokens, top_p=0.9)
        # Fix #4: stop decoding at closing tags to avoid post-JSON prose.
        try:
            config.stop_sequences = ["</tool_call>", "</final>"]
        except Exception:
            pass
        if kv_bits is not None:
            setattr(config, "kv_bits", kv_bits)
        if kv_group_size is not None:
            setattr(config, "kv_group_size", kv_group_size)
        if quantized_kv_start is not None:
            setattr(config, "quantized_kv_start", quantized_kv_start)
        enable_terminal_stream = stream_output and event_callback is None
        if stream_output:
            chunks: list[str] = []
            generated_token_count = 0
            prefill_started = time.perf_counter()
            first_token_latency: float | None = None
            decode_started: float | None = None

            if enable_terminal_stream and HAS_RICH and RICH_STDERR is not None and Live is not None:
                with Live(build_status_renderable(iteration, phase="prefill", prompt_tokens=prompt_tokens, first_token_latency=None, generated_tokens=0, elapsed=0.0, tps=0.0, raw_preview=""), console=RICH_STDERR, refresh_per_second=8, transient=False) as live:
                    stream_generator = model.generate_stream_chat(messages, config=config) if use_chat else model.generate_stream(prompt, config=config)
                    for chunk in stream_generator:
                        if should_stop is not None and should_stop():
                            return LoopResult(False, "cancelled", iteration, tool_calls, transcript)
                        text = str(chunk)
                        if decode_started is None:
                            first_token_latency = time.perf_counter() - prefill_started
                            decode_started = time.perf_counter()
                        chunks.append(text)
                        # Fix #3: sample token count every 20 chunks instead of every chunk.
                        if tokenizer is not None and len(chunks) % 20 == 0:
                            generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
                        elapsed = max((time.perf_counter() - (decode_started or time.perf_counter())), 1e-6)
                        tps = generated_token_count / elapsed if generated_token_count else 0.0
                        live.update(build_status_renderable(iteration, phase="decode", prompt_tokens=prompt_tokens, first_token_latency=first_token_latency, generated_tokens=generated_token_count, elapsed=elapsed, tps=tps, raw_preview="".join(chunks)))
                        if event_callback is not None:
                            event_callback({"type": "stream_chunk", "iteration": iteration, "chunk": text})
                    # Final accurate token count after stream ends.
                    if tokenizer is not None and chunks:
                        generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
                    total_elapsed = max((time.perf_counter() - decode_started), 1e-6) if decode_started is not None else 0.0
                    total_tps = generated_token_count / total_elapsed if generated_token_count and total_elapsed else 0.0
                    live.update(build_status_renderable(iteration, phase="decode-complete" if decode_started is not None else "prefill-no-output", prompt_tokens=prompt_tokens, first_token_latency=first_token_latency, generated_tokens=generated_token_count, elapsed=total_elapsed, tps=total_tps, raw_preview="".join(chunks)))
            else:
                prefill_stop = None
                prefill_thread = None
                emitted_prefill_summary = False
                last_decode_report = 0.0
                if enable_terminal_stream:
                    prefill_stop, prefill_thread, _ = start_prefill_indicator(iteration, prompt_tokens)
                stream_generator = model.generate_stream_chat(messages, config=config) if use_chat else model.generate_stream(prompt, config=config)
                for chunk in stream_generator:
                    if should_stop is not None and should_stop():
                        return LoopResult(False, "cancelled", iteration, tool_calls, transcript)
                    text = str(chunk)
                    if not emitted_prefill_summary:
                        if prefill_stop is not None and prefill_thread is not None:
                            prefill_stop.set()
                            prefill_thread.join(timeout=0.3)
                        first_token_latency = time.perf_counter() - prefill_started
                        if enable_terminal_stream:
                            print(f"\r[iteration {iteration}] prefill complete prompt_tokens={prompt_tokens if prompt_tokens is not None else '?'} first_token_latency={first_token_latency:.2f}s", file=sys.stderr, flush=True)
                            print(f"[iteration {iteration}] model stream:", file=sys.stderr, flush=True)
                        if event_callback is not None:
                            event_callback(
                                {
                                    "type": "prefill_complete",
                                    "iteration": iteration,
                                    "prompt_tokens": prompt_tokens,
                                    "first_token_latency": first_token_latency,
                                }
                            )
                        emitted_prefill_summary = True
                        decode_started = time.perf_counter()
                    chunks.append(text)
                    # Fix #3: sample token count every 20 chunks instead of every chunk.
                    if tokenizer is not None and len(chunks) % 20 == 0:
                        generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
                    if event_callback is not None:
                        event_callback({"type": "stream_chunk", "iteration": iteration, "chunk": text})
                    if enable_terminal_stream:
                        print(text, end="", file=sys.stderr, flush=True)
                    now = time.perf_counter()
                    if decode_started is not None and (now - last_decode_report >= 0.5):
                        elapsed = max(now - decode_started, 1e-6)
                        tps = generated_token_count / elapsed if generated_token_count else 0.0
                        if event_callback is not None:
                            event_callback(
                                {
                                    "type": "decode_stats",
                                    "iteration": iteration,
                                    "generated_tokens": generated_token_count,
                                    "elapsed": elapsed,
                                    "tps": tps,
                                }
                            )
                        if enable_terminal_stream:
                            print(f"\n[iteration {iteration}] decode generated_tokens={generated_token_count} elapsed={elapsed:.2f}s tok_s={tps:.1f}", file=sys.stderr, flush=True)
                        last_decode_report = now
                if not emitted_prefill_summary:
                    if prefill_stop is not None and prefill_thread is not None:
                        prefill_stop.set()
                        prefill_thread.join(timeout=0.3)
                if enable_terminal_stream:
                    print("", file=sys.stderr, flush=True)
                # Final accurate token count after stream ends.
                if tokenizer is not None and chunks:
                    generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
            raw = "".join(chunks).strip()
        else:
            result = model.generate_chat(messages, config=config) if use_chat else model.generate(prompt, config=config)
            if isinstance(result, dict):
                raw = str(result.get("text", "")).strip()
            else:
                raw = str(getattr(result, "text", "")).strip()
            if event_callback is not None:
                event_callback({"type": "model_output", "iteration": iteration, "text": raw})

        generated_tokens = count_tokens(tokenizer, raw) if tokenizer is not None else None
        if hasattr(model, "last_usage") and model.last_usage:
            usage = model.last_usage
            if usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                generated_tokens = usage["completion_tokens"]
        turn: dict[str, Any] = {
            "iteration": iteration,
            "raw": raw,
            "telemetry": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            },
        }

        # --- Multi-tool path (experimental) ---
        if multi_tool:
            all_calls = extract_all_tool_calls(raw)
            if len(all_calls) >= 2:
                # Normalize each call; skip if any is non-tool (plan/final).
                normalized_calls: list[dict[str, Any]] = []
                parse_error: str | None = None
                for raw_call in all_calls:
                    try:
                        norm = normalize_response(raw_call)
                        if norm["type"] != "tool":
                            parse_error = f"Multi-tool response contained non-tool block: {norm['type']}"
                            break
                        normalized_calls.append(norm)
                    except Exception as exc:
                        parse_error = str(exc)
                        break
                if parse_error or not normalized_calls:
                    turn["observations"] = [{"ok": False, "error": parse_error or "Empty multi-tool call list."}]
                    transcript.append(turn)
                    _update_tool_stats(running_tool_stats, turn)
                    if compress_observations and len(transcript) >= 2:
                        compress_turn(transcript[-2])
                    continue
                observations: list[dict[str, Any]] = []
                added_evidence_total = 0
                for call in normalized_calls:
                    if should_stop is not None and should_stop():
                        return LoopResult(False, "cancelled", iteration, tool_calls, transcript)
                    obs = run_tool(
                        call,
                        workspace,
                        skill_catalog=skill_catalog,
                        active_skill_names=set(active_skills),
                    )
                    tool_calls += 1
                    observations.append(obs)
                    update_active_skills(active_skills, call, obs)
                    added_evidence_total += update_evidence_cache(evidence_cache, call, obs)
                turn["decisions"] = normalized_calls
                turn["decision"] = normalized_calls[0]  # keep compat for reflection hints
                turn["observations"] = observations
                turn["evidence_cache_size"] = len(evidence_cache)
                turn["new_evidence"] = added_evidence_total
                no_new_evidence_streak = 0 if added_evidence_total > 0 else no_new_evidence_streak + 1
                # narrow_empty_search streak only applies to single tool calls
                narrow_empty_search_streak = 0
                _update_tool_stats(running_tool_stats, turn)
                if event_callback is not None:
                    for call, obs in zip(normalized_calls, observations):
                        event_callback({"type": "observation", "iteration": iteration, "decision": call, "observation": obs})
                    event_callback(
                        {
                            "type": "dashboard",
                            "iteration": iteration,
                            "evidences": summarize_evidence(evidence_cache),
                            "tool_stats": dict(running_tool_stats),
                            "plan": plan_state or default_plan_state(),
                        }
                    )
                if stream_output:
                    for call, obs in zip(normalized_calls, observations):
                        emit_observation_rendered(iteration, call, obs)
                transcript.append(turn)
                if compress_observations and len(transcript) >= 2:
                    compress_turn(transcript[-2])
                continue
            # Fall through to single-tool path if fewer than 2 calls found.

        # --- Single-tool / plan / final path ---
        try:
            response = normalize_response(extract_first_json_object(raw))
        except Exception:
            turn["observations"] = [{"ok": False, "error": "Output was not a valid tool_call/final JSON block."}]
            transcript.append(turn)
            _update_tool_stats(running_tool_stats, turn)
            if compress_observations and len(transcript) >= 2:
                compress_turn(transcript[-2])
            continue

        turn["decision"] = response
        if response["type"] == "plan":
            plan_state = response["plan"]
            turn["observations"] = [{"ok": True, "plan_captured": True, "plan": plan_state}]
            if event_callback is not None:
                event_callback({"type": "observation", "iteration": iteration, "decision": response, "observation": turn["observations"][-1]})
                event_callback({"type": "plan_state", "iteration": iteration, "plan": plan_state})
                _update_tool_stats(running_tool_stats, turn)
                event_callback(
                    {
                        "type": "dashboard",
                        "iteration": iteration,
                        "evidences": summarize_evidence(evidence_cache),
                        "tool_stats": dict(running_tool_stats),
                        "plan": plan_state,
                    }
                )
            if stream_output:
                emit_observation_rendered(iteration, response, turn["observations"][-1])
            transcript.append(turn)
            if compress_observations and len(transcript) >= 2:
                compress_turn(transcript[-2])
            continue

        if response["type"] == "plan_update":
            incoming_plan = response["plan"]
            if plan_state is None:
                plan_state = incoming_plan
            else:
                merged = dict(plan_state)
                merged["goal"] = incoming_plan.get("goal", merged.get("goal"))
                merged["steps"] = incoming_plan.get("steps", merged.get("steps", []))
                merged["done_when"] = incoming_plan.get("done_when", merged.get("done_when"))
                merged["completed_steps"] = incoming_plan.get("completed_steps", merged.get("completed_steps", []))
                merged["current_step"] = incoming_plan.get("current_step", merged.get("current_step"))
                plan_state = parse_plan_payload(merged)
            turn["observations"] = [{"ok": True, "plan_updated": True, "plan": plan_state}]
            if event_callback is not None:
                event_callback({"type": "observation", "iteration": iteration, "decision": response, "observation": turn["observations"][-1]})
                event_callback({"type": "plan_state", "iteration": iteration, "plan": plan_state})
                _update_tool_stats(running_tool_stats, turn)
                event_callback(
                    {
                        "type": "dashboard",
                        "iteration": iteration,
                        "evidences": summarize_evidence(evidence_cache),
                        "tool_stats": dict(running_tool_stats),
                        "plan": plan_state,
                    }
                )
            transcript.append(turn)
            if compress_observations and len(transcript) >= 2:
                compress_turn(transcript[-2])
            continue

        if response["type"] == "final":
            previous_observations = transcript[-1].get("observations", []) if transcript else []
            if any(is_blocking_failure_for_final(observation) for observation in previous_observations):
                turn["observations"] = [{"ok": False, "error": "Cannot finish immediately after a blocking tool failure."}]
                if event_callback is not None:
                    event_callback({"type": "observation", "iteration": iteration, "decision": response, "observation": turn["observations"][-1]})
                if stream_output:
                    emit_observation_rendered(iteration, response, turn["observations"][-1])
                transcript.append(turn)
                if compress_observations and len(transcript) >= 2:
                    compress_turn(transcript[-2])
                continue
            turn["observations"] = []
            turn["finish"] = response.get("finish", "unknown")
            if event_callback is not None:
                event_callback({"type": "final", "iteration": iteration, "answer": response["answer"], "finish": turn["finish"]})
            transcript.append(turn)
            return LoopResult(True, response["answer"], iteration, tool_calls, transcript)

        if should_stop is not None and should_stop():
            return LoopResult(False, "cancelled", iteration, tool_calls, transcript)

        name = response.get("name", "")
        args = response.get("arguments", {})
        if _check_doom_loop(name, args, recent_tool_calls):
            choice = _trigger_doom_loop_wait(name, args, iteration, event_callback)
            if choice == "reject":
                observation = {"ok": True, "output": "", "skipped": True}
                turn["observations"] = [observation]
                if event_callback is not None:
                    event_callback({"type": "observation", "iteration": iteration, "decision": response, "observation": observation})
                transcript.append(turn)
                if compress_observations and len(transcript) >= 2:
                    compress_turn(transcript[-2])
                continue

        observation = run_tool(
            response,
            workspace,
            skill_catalog=skill_catalog,
            active_skill_names=set(active_skills),
        )
        tool_calls += 1
        recent_tool_calls.append(ToolCallRecord(name, _build_args_frozen(args), time.time()))
        recent_tool_calls = _cleanup_expired_entries(recent_tool_calls)
        update_active_skills(active_skills, response, observation)
        turn["observations"] = [observation]
        added_evidence = update_evidence_cache(evidence_cache, response, observation)
        turn["evidence_cache_size"] = len(evidence_cache)
        turn["new_evidence"] = added_evidence
        no_new_evidence_streak = 0 if added_evidence > 0 else no_new_evidence_streak + 1
        narrow_empty_search_streak = narrow_empty_search_streak + 1 if is_empty_search_observation(response, observation) and looks_like_narrow_search(response) else 0
        _update_tool_stats(running_tool_stats, turn)
        if event_callback is not None:
            event_callback({"type": "observation", "iteration": iteration, "decision": response, "observation": observation})
            event_callback(
                {
                    "type": "dashboard",
                    "iteration": iteration,
                    "evidences": summarize_evidence(evidence_cache),
                    "tool_stats": dict(running_tool_stats),
                    "plan": plan_state or default_plan_state(),
                }
            )
        if stream_output:
            emit_observation_rendered(iteration, response, observation)
        transcript.append(turn)
        if compress_observations and len(transcript) >= 2:
            compress_turn(transcript[-2])
        if narrow_empty_search_streak >= 3:
            injected = {"iteration": iteration, "raw": "<controller_observation>", "observations": [{"ok": False, "controller_guidance": True, "error": "Repeated narrow search returned no results. Broaden the search: enumerate a parent directory, inspect candidate files already found, or use a wider pattern before guessing more exact names."}], "controller_injected_observation": True}
            if event_callback is not None:
                event_callback({"type": "observation", "iteration": iteration, "decision": {"type": "controller", "name": "recovery"}, "observation": injected["observations"][-1]})
            if stream_output:
                emit_observation_rendered(iteration, {"type": "controller", "name": "recovery"}, injected["observations"][-1])
            transcript.append(injected)
            narrow_empty_search_streak = 0

    return LoopResult(False, "max_iterations_reached", max_iterations, tool_calls, transcript)
