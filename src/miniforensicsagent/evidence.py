from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EvidenceRubric:
    strong_evidence: list[str]
    weak_evidence: list[str]
    finish_when: str


UNTRUSTED_PATH_PARTS = {"backups", "backup", "docs", "doc", "examples", "example", "tests", "test", "fixtures", "fixture", "samples", "sample"}
DECOY_TEXT_MARKERS = {"example", "mock", "fixture", "fallback", "retired", "legacy", "skipped", "dummy", "placeholder"}
SUCCESS_TEXT_MARKERS = {"status=success", '"status":"success"', "used_", "executed", "completed"}
USAGE_TEXT_MARKERS = {"fetch(", "headers", "authorization", "x-api-key", "bearer", "runtimeconfig.", "client", "request", "process.env.", "postgres("}


def is_trusted_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    parts = {part.lower() for part in normalized.split("/") if part}
    return not any(part in UNTRUSTED_PATH_PARTS for part in parts)


def classify_line_role(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in SUCCESS_TEXT_MARKERS):
        return "success"
    if any(marker in lowered for marker in USAGE_TEXT_MARKERS):
        return "usage"
    return "value"


def line_is_decoy(path: str, text: str) -> bool:
    lowered = text.lower()
    return (not is_trusted_path(path)) or any(marker in lowered for marker in DECOY_TEXT_MARKERS)


def extract_candidate_values(text: str) -> set[str]:
    candidates: set[str] = set()
    for value in re.findall(r'[:=]\s*"([^"]{3,120})"', text):
        stripped = value.strip()
        if stripped and " " not in stripped and not stripped.startswith(("http", "@")) and not stripped.endswith((".ts", ".js", ".json", ".md")) and any(char.isalpha() for char in stripped) and (any(char.isdigit() for char in stripped) or any(char in ":-._" for char in stripped)):
            candidates.add(stripped)
    for value in re.findall(r"\b(pk_[A-Za-z0-9:_-]+)\b", text):
        candidates.add(value)
    for value in re.findall(r"\bprocess\.env\.([A-Z][A-Z0-9_]{2,})\b", text):
        candidates.add(value)
    for value in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", text):
        if value not in {"GET", "POST", "PUT", "DELETE", "PATCH", "UUID", "ASC", "DESC"} and value.count("_") >= 1:
            candidates.add(value)
    return candidates


def extract_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for symbol in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b", text):
        if symbol.lower() in {"true", "false", "null", "return", "const", "type"}:
            continue
        symbols.add(symbol)
        if "." in symbol:
            symbols.add(symbol.split(".")[-1])
    return symbols


def update_evidence_cache(cache: dict[str, dict[str, Any]], decision: dict[str, Any], observation: dict[str, Any]) -> int:
    added = 0
    if not observation.get("ok"):
        return added
    symbol_cache = cache.setdefault("__symbols__", {"artifact_type": "symbol_index", "symbols": {}})

    def record_line(path: str, line: str) -> None:
        nonlocal added
        role = classify_line_role(line)
        trusted = not line_is_decoy(path, line)
        candidates = extract_candidate_values(line)
        symbols = extract_symbols(line)
        for symbol in symbols:
            bucket = symbol_cache["symbols"].setdefault(symbol, {"usage": 0, "trusted_usage": 0, "paths": set()})
            if role == "usage":
                bucket["usage"] += 1
                if trusted:
                    bucket["trusted_usage"] += 1
            bucket["paths"].add(path)
        for candidate in candidates:
            bucket = cache.setdefault(candidate, {"artifact_type": "candidate", "sources": set(), "evidence": [], "symbols": set()})
            key = (role, trusted, path, line)
            if key in bucket["sources"]:
                continue
            bucket["sources"].add(key)
            bucket["symbols"].update(symbols)
            bucket["evidence"].append({"role": role, "trusted": trusted, "path": path, "text": line})
            added += 1

    if decision["name"] == "Read":
        path = str(decision["arguments"].get("file_path", ""))
        for raw_line in str(observation.get("content", "")).splitlines():
            record_line(path, raw_line.strip())
    if decision["name"] == "Grep":
        for match in observation.get("matches", []):
            record_line(str(match.get("file", "")), str(match.get("text", "")))
    return added


def default_evidence_rubric() -> EvidenceRubric:
    return EvidenceRubric(
        strong_evidence=["direct runtime usage in executable code", "multiple trusted code references"],
        weak_evidence=["docs or comments", "helper scripts or candidate lists"],
        finish_when="finish when one candidate is supported by direct trusted usage and alternatives remain weak",
    )


def parse_evidence_rubric(payload: dict[str, Any]) -> EvidenceRubric:
    fallback = default_evidence_rubric()
    strong = payload.get("strong_evidence")
    weak = payload.get("weak_evidence")
    finish_when = str(payload.get("finish_when", "")).strip()
    if not isinstance(strong, list) or not strong:
        strong = fallback.strong_evidence
    if not isinstance(weak, list) or not weak:
        weak = fallback.weak_evidence
    if not finish_when:
        finish_when = fallback.finish_when
    return EvidenceRubric(
        strong_evidence=[str(item) for item in strong[:4]],
        weak_evidence=[str(item) for item in weak[:4]],
        finish_when=finish_when,
    )


def has_promising_but_incomplete_candidate(cache: dict[str, dict[str, Any]]) -> bool:
    symbol_index = cache.get("__symbols__", {}).get("symbols", {})
    for value, payload in cache.items():
        if value == "__symbols__" or payload.get("artifact_type") != "candidate":
            continue
        trusted_value_hits = sum(1 for item in payload["evidence"] if item["trusted"] and item["role"] == "value")
        trusted_success_hits = sum(1 for item in payload["evidence"] if item["trusted"] and item["role"] == "success")
        linked_usage_hits = sum(int(symbol_index.get(symbol, {}).get("trusted_usage", 0)) for symbol in payload.get("symbols", set()))
        if trusted_value_hits > 0 and trusted_success_hits == 0 and linked_usage_hits == 0:
            return True
    return False
