from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import (
    EvidenceRubric,
    default_evidence_rubric,
    has_promising_but_incomplete_candidate,
    parse_evidence_rubric,
    update_evidence_cache,
)
from .prompting import build_prompt
from .render import HAS_RICH, Live, RICH_STDERR, build_status_renderable, count_tokens, emit_observation_rendered, start_prefill_indicator
from .tools import DEFAULT_READ_LIMIT, run_tool


TOOL_NAMES = {"Read", "Glob", "Grep", "Bash", "Write", "Edit"}


@dataclass
class LoopResult:
    success: bool
    answer: str
    iterations: int
    tool_calls: int
    transcript: list[dict[str, Any]]


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


def normalize_response(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("type") == "rubric" or {"strong_evidence", "weak_evidence", "finish_when"} <= set(payload.keys()):
        parsed = parse_evidence_rubric(payload)
        return {"type": "rubric", "strong_evidence": parsed.strong_evidence, "weak_evidence": parsed.weak_evidence, "finish_when": parsed.finish_when}
    if payload.get("type") == "final" or payload.get("action") == "final":
        return {"type": "final", "answer": str(payload.get("answer", payload.get("summary", "done")))}
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


def same_read_window(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("type") != "tool" or right.get("type") != "tool" or left.get("name") != "Read" or right.get("name") != "Read":
        return False
    left_args = left.get("arguments", {})
    right_args = right.get("arguments", {})
    return str(left_args.get("file_path", "")) == str(right_args.get("file_path", "")) and int(left_args.get("offset", left_args.get("start_line", 1))) == int(right_args.get("offset", right_args.get("start_line", 1))) and int(left_args.get("limit", left_args.get("max_lines", DEFAULT_READ_LIMIT))) == int(right_args.get("limit", right_args.get("max_lines", DEFAULT_READ_LIMIT)))


def expanding_read_from_start(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("type") != "tool" or right.get("type") != "tool" or left.get("name") != "Read" or right.get("name") != "Read":
        return False
    left_args = left.get("arguments", {})
    right_args = right.get("arguments", {})
    if str(left_args.get("file_path", "")) != str(right_args.get("file_path", "")):
        return False
    if int(left_args.get("offset", left_args.get("start_line", 1))) != 1 or int(right_args.get("offset", right_args.get("start_line", 1))) != 1:
        return False
    return int(left_args.get("limit", left_args.get("max_lines", DEFAULT_READ_LIMIT))) > int(right_args.get("limit", right_args.get("max_lines", DEFAULT_READ_LIMIT)))


def suggested_window_from_recent_grep(transcript: list[dict[str, Any]], read_call: dict[str, Any]) -> tuple[int, int] | None:
    target = str(read_call.get("arguments", {}).get("file_path", ""))
    for turn in reversed(transcript):
        if turn.get("decision", {}).get("name") != "Grep":
            continue
        matches = turn.get("observations", [{}])[-1].get("matches", [])
        if not isinstance(matches, list):
            continue
        for match in matches:
            if str(match.get("file", "")) == target:
                return max(1, int(match.get("line", 1)) - 10), 30
    return None


def recent_alternative_files_from_grep(transcript: list[dict[str, Any]], current_file: str, *, limit: int = 3) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()
    for turn in reversed(transcript):
        if turn.get("decision", {}).get("name") != "Grep":
            continue
        matches = turn.get("observations", [{}])[-1].get("matches", [])
        if not isinstance(matches, list):
            continue
        for match in matches:
            path = str(match.get("file", ""))
            if path and path != current_file and path not in seen:
                seen.add(path)
                suggestions.append(path)
                if len(suggestions) >= limit:
                    return suggestions
    return suggestions


def suggested_read_for_file_from_recent_grep(transcript: list[dict[str, Any]], file_path: str) -> dict[str, Any] | None:
    hinted_window = suggested_window_from_recent_grep(transcript, {"arguments": {"file_path": file_path}})
    if not hinted_window:
        return None
    offset, limit = hinted_window
    return {"file_path": file_path, "offset": offset, "limit": limit}


def repeated_read_same_file(transcript: list[dict[str, Any]], current_read: dict[str, Any], *, threshold: int = 3) -> bool:
    target = str(current_read.get("arguments", {}).get("file_path", ""))
    repeats = 0
    for turn in reversed(transcript):
        if turn.get("decision", {}).get("name") != "Read":
            continue
        if str(turn.get("decision", {}).get("arguments", {}).get("file_path", "")) == target:
            repeats += 1
            if repeats >= threshold:
                return True
    return False


def is_blocking_failure_for_final(observation: dict[str, Any]) -> bool:
    if observation.get("ok", True):
        return False
    return not observation.get("controller_guidance", False)


def run_loop(model: Any, generation_config: Any, *, task: str, workspace: Path, max_iterations: int, max_tokens: int, temperature: float, stream_output: bool, reflection_strength: str, kv_bits: int | None = None, kv_group_size: int | None = None, quantized_kv_start: int | None = None) -> LoopResult:
    transcript: list[dict[str, Any]] = []
    tool_calls = 0
    evidence_cache: dict[str, dict[str, Any]] = {}
    no_new_evidence_streak = 0
    narrow_empty_search_streak = 0
    rubric: EvidenceRubric | None = None

    for iteration in range(1, max_iterations + 1):
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

        prompt = build_prompt(task, transcript, workspace, rubric=rubric or default_evidence_rubric(), remaining_iterations=max_iterations - iteration + 1, reflection_hint=reflection_hint)
        config = generation_config(temperature=temperature, max_tokens=max_tokens, top_p=0.9)
        if kv_bits is not None:
            setattr(config, "kv_bits", kv_bits)
        if kv_group_size is not None:
            setattr(config, "kv_group_size", kv_group_size)
        if quantized_kv_start is not None:
            setattr(config, "quantized_kv_start", quantized_kv_start)
        if stream_output:
            tokenizer = getattr(model, "tokenizer", None)
            prompt_tokens = count_tokens(tokenizer, prompt) if tokenizer is not None else None
            chunks: list[str] = []
            generated_token_count = 0
            prefill_started = time.perf_counter()
            first_token_latency: float | None = None
            decode_started: float | None = None

            if HAS_RICH and RICH_STDERR is not None and Live is not None:
                with Live(build_status_renderable(iteration, phase="prefill", prompt_tokens=prompt_tokens, first_token_latency=None, generated_tokens=0, elapsed=0.0, tps=0.0, raw_preview=""), console=RICH_STDERR, refresh_per_second=8, transient=False) as live:
                    for chunk in model.generate_stream(prompt, config=config):
                        text = str(chunk)
                        if decode_started is None:
                            first_token_latency = time.perf_counter() - prefill_started
                            decode_started = time.perf_counter()
                        chunks.append(text)
                        if tokenizer is not None:
                            generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
                        elapsed = max((time.perf_counter() - (decode_started or time.perf_counter())), 1e-6)
                        tps = generated_token_count / elapsed if generated_token_count else 0.0
                        live.update(build_status_renderable(iteration, phase="decode", prompt_tokens=prompt_tokens, first_token_latency=first_token_latency, generated_tokens=generated_token_count, elapsed=elapsed, tps=tps, raw_preview="".join(chunks)))
                    total_elapsed = max((time.perf_counter() - decode_started), 1e-6) if decode_started is not None else 0.0
                    total_tps = generated_token_count / total_elapsed if generated_token_count and total_elapsed else 0.0
                    live.update(build_status_renderable(iteration, phase="decode-complete" if decode_started is not None else "prefill-no-output", prompt_tokens=prompt_tokens, first_token_latency=first_token_latency, generated_tokens=generated_token_count, elapsed=total_elapsed, tps=total_tps, raw_preview="".join(chunks)))
            else:
                prefill_stop, prefill_thread, _ = start_prefill_indicator(iteration, prompt_tokens)
                emitted_prefill_summary = False
                last_decode_report = 0.0
                for chunk in model.generate_stream(prompt, config=config):
                    text = str(chunk)
                    if not emitted_prefill_summary:
                        prefill_stop.set()
                        prefill_thread.join(timeout=0.3)
                        first_token_latency = time.perf_counter() - prefill_started
                        print(f"\r[iteration {iteration}] prefill complete prompt_tokens={prompt_tokens if prompt_tokens is not None else '?'} first_token_latency={first_token_latency:.2f}s", file=sys.stderr, flush=True)
                        print(f"[iteration {iteration}] model stream:", file=sys.stderr, flush=True)
                        emitted_prefill_summary = True
                        decode_started = time.perf_counter()
                    chunks.append(text)
                    if tokenizer is not None:
                        generated_token_count = count_tokens(tokenizer, "".join(chunks)) or generated_token_count
                    print(text, end="", file=sys.stderr, flush=True)
                    now = time.perf_counter()
                    if decode_started is not None and (now - last_decode_report >= 0.5):
                        elapsed = max(now - decode_started, 1e-6)
                        tps = generated_token_count / elapsed if generated_token_count else 0.0
                        print(f"\n[iteration {iteration}] decode generated_tokens={generated_token_count} elapsed={elapsed:.2f}s tok_s={tps:.1f}", file=sys.stderr, flush=True)
                        last_decode_report = now
                if not emitted_prefill_summary:
                    prefill_stop.set()
                    prefill_thread.join(timeout=0.3)
                print("", file=sys.stderr, flush=True)
            raw = "".join(chunks).strip()
        else:
            raw = getattr(model.generate(prompt, config=config), "text", "").strip()

        turn: dict[str, Any] = {"iteration": iteration, "raw": raw}
        try:
            response = normalize_response(extract_first_json_object(raw))
        except Exception:
            turn["observations"] = [{"ok": False, "error": "Output was not a valid tool_call/final JSON block."}]
            transcript.append(turn)
            continue

        turn["decision"] = response
        if response["type"] == "rubric":
            rubric = parse_evidence_rubric(response)
            turn["observations"] = [{"ok": True, "rubric_captured": True}]
            if stream_output:
                emit_observation_rendered(iteration, response, turn["observations"][-1])
            transcript.append(turn)
            continue

        if response["type"] == "final":
            previous_observations = transcript[-1].get("observations", []) if transcript else []
            if any(is_blocking_failure_for_final(observation) for observation in previous_observations):
                turn["observations"] = [{"ok": False, "error": "Cannot finish immediately after a blocking tool failure."}]
                if stream_output:
                    emit_observation_rendered(iteration, response, turn["observations"][-1])
                transcript.append(turn)
                continue
            turn["observations"] = []
            transcript.append(turn)
            return LoopResult(True, response["answer"], iteration, tool_calls, transcript)

        previous_decision = transcript[-1].get("decision", {}) if transcript else {}
        if same_read_window(response, previous_decision):
            turn["observations"] = [{"ok": False, "controller_guidance": True, "error": "The same Read window was requested again. Change strategy now. Change strategy now. Change strategy now. Use a different offset/limit, inspect a different file, or use Grep to locate a more precise line before reading."}]
            if stream_output:
                emit_observation_rendered(iteration, response, turn["observations"][-1])
            transcript.append(turn)
            continue
        if expanding_read_from_start(response, previous_decision):
            turn["observations"] = [{"ok": False, "controller_guidance": True, "error": "Do not expand the same file from offset=1. Use Grep to locate a line, then Read a small window around that line, or switch to a different file."}]
            if stream_output:
                emit_observation_rendered(iteration, response, turn["observations"][-1])
            transcript.append(turn)
            continue
        if response.get("type") == "tool" and response.get("name") == "Read":
            hinted_window = suggested_window_from_recent_grep(transcript, response)
            read_args = response.get("arguments", {})
            target_file = str(read_args.get("file_path", ""))
            requested_offset = int(read_args.get("offset", read_args.get("start_line", 1)))
            requested_limit = int(read_args.get("limit", read_args.get("max_lines", DEFAULT_READ_LIMIT)))
            if repeated_read_same_file(transcript, response):
                alternative_files = recent_alternative_files_from_grep(transcript, target_file)
                suggested_read = suggested_read_for_file_from_recent_grep(transcript, alternative_files[0]) if alternative_files else None
                turn["observations"] = [{"ok": False, "controller_guidance": True, "error": "This file has already been read multiple times. Switch to a different file unless you have a very specific new offset to inspect.", "switch_file_suggestions": alternative_files, "suggested_read": suggested_read}]
                if stream_output:
                    emit_observation_rendered(iteration, response, turn["observations"][-1])
                transcript.append(turn)
                continue
            if hinted_window and (requested_offset, requested_limit) != hinted_window:
                offset, limit = hinted_window
                turn["observations"] = [{"ok": False, "controller_guidance": True, "error": f"A recent Grep already provided a relevant line. Read around that line instead, for example offset={offset}, limit={limit}."}]
                if stream_output:
                    emit_observation_rendered(iteration, response, turn["observations"][-1])
                transcript.append(turn)
                continue

        observation = run_tool(response, workspace)
        tool_calls += 1
        turn["observations"] = [observation]
        added_evidence = update_evidence_cache(evidence_cache, response, observation)
        turn["evidence_cache_size"] = len(evidence_cache)
        turn["new_evidence"] = added_evidence
        no_new_evidence_streak = 0 if added_evidence > 0 else no_new_evidence_streak + 1
        narrow_empty_search_streak = narrow_empty_search_streak + 1 if is_empty_search_observation(response, observation) and looks_like_narrow_search(response) else 0
        if stream_output:
            emit_observation_rendered(iteration, response, observation)
        transcript.append(turn)
        if narrow_empty_search_streak >= 3:
            injected = {"iteration": iteration, "raw": "<controller_observation>", "observations": [{"ok": False, "controller_guidance": True, "error": "Repeated narrow search returned no results. Broaden the search: enumerate a parent directory, inspect candidate files already found, or use a wider pattern before guessing more exact names."}], "controller_injected_observation": True}
            if stream_output:
                emit_observation_rendered(iteration, {"type": "controller", "name": "recovery"}, injected["observations"][-1])
            transcript.append(injected)
            narrow_empty_search_streak = 0

    return LoopResult(False, "max_iterations_reached", max_iterations, tool_calls, transcript)
