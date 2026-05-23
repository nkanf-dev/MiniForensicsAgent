from __future__ import annotations

import json
from pathlib import Path

def build_prompt(
    task: str,
    transcript: list[dict],
    workspace: Path,
    *,
    current_plan: dict | None = None,
    remaining_iterations: int | None = None,
    reflection_hint: str = "",
    window: int | None = None,
    multi_tool: bool = False,
    available_skills: str = "",
    active_skill_context: str = "",
) -> str:
    visible = transcript[-window:] if window is not None and window > 0 else transcript
    history = json.dumps(visible, ensure_ascii=False, indent=2) if visible else "[]"
    few_shot = """Examples:
{"type":"plan","goal":"find one config value","steps":["discover files","locate runtime usage","verify evidence"],"done_when":"one value is proven by runtime usage"}
<tool_call>{"name":"Glob","arguments":{"pattern":"**/*.js","path":"."}}</tool_call>
<tool_call>{"name":"Grep","arguments":{"pattern":"apiKey","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/index.ts","offset":48,"limit":24}}</tool_call>
{"type":"plan_update","completed_steps":["discover files"],"current_step":"locate runtime usage"}
<tool_call>{"name":"Grep","arguments":{"pattern":"targetSymbol","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/module.ts","offset":22,"limit":30}}</tool_call>
<final>{"answer":"found the value and verified how it is used"}</final>

Final answer examples - ALWAYS use <final> tag:
<final>{"answer":"The config value is 'debug mode' found in config.py line 42"}</final>
<final>{"answer":"2+2 equals 4"}</final>
<final>{"answer":"API endpoint is /api/v1/users based on route definition"}</final>

When you have completed the task and found the answer, you MUST output:
<final>{"answer":"your answer here"}</final>
Do NOT output plain text. Do NOT output explanations before <final>."""
    plan_block = ""
    if current_plan:
        plan_block = (
            "Current plan:\n"
            + json.dumps(current_plan, ensure_ascii=False, indent=2)
            + "\n"
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
    plan_instruction = ""
    if transcript == []:
        plan_instruction = """First turn requirement:
Return a plan JSON object and nothing else:
{"type":"plan","goal":"short sentence","steps":["short step","short step"],"done_when":"one short sentence"}
Do not call a tool on the first turn.
"""
    if multi_tool:
        tool_call_rule = (
            "- You may issue one OR MORE independent tool calls in a single turn by emitting multiple <tool_call> blocks.\n"
            "- Only batch tool calls that are truly independent (e.g. Grep + Glob). Do not batch a Read that depends on a Grep result.\n"
            "- Do not mix tool calls with plan/final in the same turn."
        )
        turn_format = (
            "Return one or more tool calls, OR exactly one plan/plan_update/final — never mixed:\n"
            "- <tool_call>{\"name\":\"Glob\",\"arguments\":{\"pattern\":\"**/*.js\",\"path\":\".\"}}}</tool_call>\n"
            "  <tool_call>{\"name\":\"Grep\",\"arguments\":{\"pattern\":\"apiKey\",\"path\":\".\"}}}</tool_call>\n"
            "- or {\"type\":\"plan\",\"goal\":\"short sentence\",\"steps\":[\"step\"],\"done_when\":\"one short sentence\"}\n"
            "- or {\"type\":\"plan_update\",\"completed_steps\":[\"step\"],\"current_step\":\"step\"}\n"
            "- or <final>{\"answer\":\"done\"}</final>"
        )
    else:
        tool_call_rule = "- One tool call per turn."
        turn_format = (
            "Return exactly one thing each turn:\n"
            "- <tool_call>{\"name\":\"Glob\",\"arguments\":{\"pattern\":\"**/*.js\",\"path\":\".\"}}}</tool_call>\n"
            "- or {\"type\":\"plan\",\"goal\":\"short sentence\",\"steps\":[\"step\"],\"done_when\":\"one short sentence\"}\n"
            "- or {\"type\":\"plan_update\",\"completed_steps\":[\"step\"],\"current_step\":\"step\"}\n"
            "- or <final>{\"answer\":\"done\"}</final>"
        )
    skill_block = ""
    skill_tools = ""
    skill_rules = ""
    if available_skills:
        skill_block = (
            "Available skills:\n"
            f"{available_skills}\n"
        )
        skill_tools = (
            "- ActivateSkill(skill_name)\n"
            "- ReadSkillResource(skill_name, file_path, offset=1, limit=120)\n"
        )
        skill_rules = (
            "Skills:\n"
            "- If one of the listed skills clearly matches the task, activate it before following its detailed instructions.\n"
            "- Once a skill is activated, treat its instructions as active guidance for the rest of the run.\n"
            "- Only read skill resources after activating that skill, and prefer the specific files the skill references.\n"
        )
    active_skill_block = ""
    if active_skill_context:
        active_skill_block = f"{active_skill_context}\n\n"
    return f"""You are a local codebase explorer.
Use Claude Code style tool calls.
{turn_format}
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
{skill_tools}

Forensics mode:
- Treat the workspace as an evidence snapshot, not a live system.
- Prefer direct evidence from executable code, runtime references, logs, and persisted config over helper scripts, docs, comments, and examples.
- Distinguish weak clues from strong proof.
- Keep the plan updated and use it to decide next actions.

Rules:
- Use relative paths.
{tool_call_rule}
- Read is paged by default. If you need more, read another range instead of assuming the whole file is visible.
- Prefer Glob to discover files, Grep to locate lines, then Read a small window around the relevant line.
- After a Grep hit, prefer Read(file_path=matched file, offset=matched line - 10, limit=30).
- Do not default Read to offset=1 when Grep already identified a relevant line.
- Do not keep increasing Read from offset=1 unless no line-targeted option exists.
- If the last observation failed, fix it instead of finishing.
- Goal is artifact discovery, not long explanation.
{skill_rules}

{plan_block}
{convergence_block}
{final_only_prefix}
{plan_instruction}
{few_shot}
{skill_block}{active_skill_block}{reflection_hint}

Task:
{task}

Previous turns:
{history}
"""


def build_chat_messages(
    task: str,
    transcript: list[dict],
    workspace: Path,
    *,
    current_plan: dict | None = None,
    remaining_iterations: int | None = None,
    reflection_hint: str = "",
    multi_tool: bool = False,
    available_skills: str = "",
    active_skill_context: str = "",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    system_content = """You are a local codebase explorer.
Use Claude Code style tool calls.
Return exactly one thing each turn:
- <tool_call>{"name":"Glob","arguments":{"pattern":"**/*.py","path":"."}}</tool_call>
- or {"type":"plan","goal":"short sentence","steps":["step"],"done_when":"one short sentence"}
- or {"type":"plan_update","completed_steps":["step"],"current_step":"step"}
- or <final>{"answer":"done"}</final>
No prose.

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
- Keep the plan updated and use it to decide next actions.

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

First turn requirement:
Return a plan JSON object and nothing else:
{"type":"plan","goal":"short sentence","steps":["short step","short step"],"done_when":"one short sentence"}
Do not call a tool on the first turn.

Examples:
{"type":"plan","goal":"find one config value","steps":["discover files","locate runtime usage","verify evidence"],"done_when":"one value is proven by runtime usage"}
<tool_call>{"name":"Glob","arguments":{"pattern":"**/*.js","path":"."}}</tool_call>
<tool_call>{"name":"Grep","arguments":{"pattern":"apiKey","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/index.ts","offset":48,"limit":24}}</tool_call>
{"type":"plan_update","completed_steps":["discover files"],"current_step":"locate runtime usage"}
<tool_call>{"name":"Grep","arguments":{"pattern":"targetSymbol","path":".","glob":"**/*"}}</tool_call>
<tool_call>{"name":"Read","arguments":{"file_path":"src/module.ts","offset":22,"limit":30}}</tool_call>
<final>{"answer":"found the value and verified how it is used"}</final>

Final answer examples - ALWAYS use <final> tag:
<final>{"answer":"The config value is 'debug mode' found in config.py line 42"}</final>
<final>{"answer":"2+2 equals 4"}</final>
<final>{"answer":"API endpoint is /api/v1/users based on route definition"}</final>

When you have completed the task and found the answer, you MUST output:
<final>{"answer":"your answer here"}</final>
Do NOT output plain text. Do NOT output explanations before <final>."""

    messages.append({"role": "system", "content": system_content})

    for turn in transcript:
        raw = turn.get("raw", "")
        decision = turn.get("decision", {})
        observations = turn.get("observations", [])

        if decision.get("type") == "tool":
            args = decision.get("arguments", {})
            tool_name = decision.get("name", "")
            tool_call_str = f'<tool_call>{{"name":"{tool_name}","arguments":{json.dumps(args)}}}</tool_call>'
            messages.append({"role": "assistant", "content": tool_call_str})

            for obs in observations:
                if obs.get("ok"):
                    if obs.get("content"):
                        content = obs["content"]
                    else:
                        safe_obs = {k: v for k, v in obs.items() if k not in ("hint",)}
                        content = json.dumps(safe_obs, ensure_ascii=False)
                    if len(content) > 2000:
                        content = content[:2000] + f"\n... [truncated {len(content)-2000} chars]"
                    messages.append({"role": "tool", "content": content})
                else:
                    messages.append({"role": "tool", "content": f"Error: {obs.get('error')}"})

        elif decision.get("type") == "final":
            messages.append({"role": "assistant", "content": raw})

        elif decision.get("type") in ("plan", "plan_update"):
            messages.append({"role": "assistant", "content": raw})

    user_parts: list[str] = []

    if current_plan:
        user_parts.append(f"Current plan:\n{json.dumps(current_plan, ensure_ascii=False)}")

    if remaining_iterations is not None and remaining_iterations <= 2:
        user_parts.append(f"[Warning: Only {remaining_iterations} iteration{'s' if remaining_iterations > 1 else ''} remaining]")

    user_parts.append(f"Task: {task}")

    if reflection_hint:
        user_parts.append(f"\nReflection: {reflection_hint}")

    if available_skills:
        user_parts.append(f"\nAvailable skills:\n{available_skills}")

    if active_skill_context:
        user_parts.append(f"\nActive skill context:\n{active_skill_context}")

    messages.append({"role": "user", "content": "\n".join(user_parts)})

    return messages
