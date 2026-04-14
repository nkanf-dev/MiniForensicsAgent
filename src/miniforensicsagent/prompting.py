from __future__ import annotations

import json
from pathlib import Path

from .evidence import EvidenceRubric


def build_prompt(task: str, transcript: list[dict], workspace: Path, *, rubric: EvidenceRubric | None = None, remaining_iterations: int | None = None, reflection_hint: str = "") -> str:
    history = json.dumps(transcript, ensure_ascii=False, indent=2) if transcript else "[]"
    few_shot = """Examples:
<tool_call>{"name":"Glob","arguments":{"pattern":"**/*.js","path":"."}}</tool_call>
<tool_call>{"name":"Grep","arguments":{"pattern":"apiKey","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/index.ts","offset":48,"limit":24}}</tool_call>
<tool_call>{"name":"Grep","arguments":{"pattern":"targetSymbol","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/module.ts","offset":22,"limit":30}}</tool_call>
<final>{"answer":"found the value and verified how it is used"}</final>"""
    rubric_block = ""
    if rubric:
        rubric_block = (
            "Evidence rubric:\n"
            f"- Strong evidence: {'; '.join(rubric.strong_evidence)}\n"
            f"- Weak evidence: {'; '.join(rubric.weak_evidence)}\n"
            f"- Finish when: {rubric.finish_when}\n"
        )
    convergence_block = ""
    final_only_prefix = ""
    if remaining_iterations is not None and remaining_iterations <= 2:
        if remaining_iterations == 1:
            convergence_block = (
                "Final decision window:\n"
                "- This is the last iteration.\n"
                "- You MUST return <final> with an analysis conclusion based on the best trusted evidence you already have.\n"
                "- Do not call another tool on the last iteration.\n"
                "- Explain which candidate is most likely correct and why weaker clues were rejected.\n"
            )
            final_only_prefix = (
                "LAST ITERATION REQUIREMENT:\n"
                "- Return <final> now.\n"
                "- Return <final> now.\n"
                "- Do not use any tool.\n"
            )
        else:
            convergence_block = (
                "Final decision window:\n"
                "- You are near the iteration limit.\n"
                "- Either finish now with the best trusted answer, or do exactly one highest-value verification action.\n"
                "- Do not keep rereading the same file from offset=1.\n"
            )
    rubric_instruction = ""
    if transcript == []:
        rubric_instruction = """First turn requirement:
Return a rubric JSON object and nothing else:
{"type":"rubric","strong_evidence":["short item"],"weak_evidence":["short item"],"finish_when":"one short sentence"}
Do not call a tool on the first turn.
"""
    return f"""You are a local codebase explorer.
Use Claude Code style tool calls.
Return exactly one thing each turn:
- <tool_call>{{"name":"Glob","arguments":{{"pattern":"**/*.js","path":"."}}}}</tool_call>
- or {{"type":"rubric","strong_evidence":["short item"],"weak_evidence":["short item"],"finish_when":"one short sentence"}}
- or <final>{{"answer":"done"}}</final>
No prose.

{final_only_prefix}

Workspace: {workspace}
Tools:
- Read(file_path, offset=1, limit=80)
- Glob(pattern, path=".")
- Grep(pattern, path=".")
- Bash(command) [allowed: pwd, ls, find, cat]
- Write(file_path, content)
- Edit(file_path, old_string, new_string)

Forensics mode:
- Treat the workspace as an evidence snapshot, not a live system.
- Prefer direct evidence from executable code, runtime references, logs, and persisted config over helper scripts, docs, comments, and examples.
- Distinguish weak clues from strong proof.
- Finish when one candidate is supported by trusted usage evidence and alternatives are only weak clues.

Rules:
- Use relative paths.
- One tool call per turn.
- Read is paged by default. If you need more, read another range instead of assuming the whole file is visible.
- Prefer Glob to discover files, Grep to locate lines, then Read a small window around the relevant line.
- After a Grep hit, prefer Read(file_path=matched file, offset=matched line - 10, limit=30).
- Do not default Read to offset=1 when Grep already identified a relevant line.
- Do not keep increasing Read from offset=1 unless no line-targeted option exists.
- If the last observation failed, fix it instead of finishing.
- Goal is artifact discovery, not long explanation.

{rubric_block}
{convergence_block}
{final_only_prefix}
{rubric_instruction}
{few_shot}
{reflection_hint}

Task:
{task}

Previous turns:
{history}
"""
