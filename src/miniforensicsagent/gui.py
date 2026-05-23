"""NiceGUI-based graphical interface for MiniForensicsAgent."""

from __future__ import annotations

import json
import queue as _queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from nicegui import ui

from .loop import run_loop
from .models import (
    DEFAULT_AGENT_WORKSPACE,
    DEFAULT_MODEL_ROOT,
    discover_models,
    load_local_model,
    patch_mlx_lm_prompt_cache_with_turboquant,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GuiConfig:
    model: str = "LocoOperator"
    model_root: str = str(DEFAULT_MODEL_ROOT)
    engine: str = "mlx"
    llama_cpp_url: str = "http://localhost:8080/v1"
    llama_cpp_model: str = "qwen3.5-9b-instruct.Q4_K_M_deepseek4.gguf"
    workspace: str = str(DEFAULT_AGENT_WORKSPACE)
    max_iterations: int = 12
    max_tokens: int = 768
    temperature: float = 0.3
    reflection_strength: str = "medium"
    kv_bits: int | None = None
    kv_group_size: int | None = None
    quantized_kv_start: int | None = None
    turboquant: bool = False
    tq_r_bits: int = 4
    tq_theta_bits: int = 4
    use_chat: bool = False
    # Experimental
    compress_observations: bool = False
    transcript_window: int | None = None
    multi_tool: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _fmt_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
    except Exception:
        return ts


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    return None if raw == "" else int(raw)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class SessionState:
    def __init__(self) -> None:
        self.config = GuiConfig()
        self.conversation_file = Path.cwd() / ".mini_forensics_conversations.json"
        self.conversations: list[dict[str, Any]] = []
        self.current_conv_index: int = 0
        self.models: list[Any] = []
        self.loaded_model_path: str | None = None
        self.loaded_engine_key: str | None = None
        self.loaded_model: Any = None
        self.generation_config_factory: Any = None
        self.running: bool = False
        self.cancel_requested: bool = False
        self.metrics: dict[str, Any] = {
            "iteration": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "ttft": 0.0,
            "tps": 0.0,
        }
        self.dashboard: dict[str, Any] = {
            "evidences": [],
            "tool_stats": {"counts": {}, "failures": 0},
            "plan": {},
        }
        self.stream_buffer: str = ""
        # Queue of zero-arg callables — worker thread enqueues, main thread executes
        self.ui_queue: _queue.SimpleQueue[Callable[[], None]] = _queue.SimpleQueue()

    # -- persistence ---------------------------------------------------------

    def load_conversations(self) -> None:
        self.conversations = []
        if not self.conversation_file.exists():
            return
        try:
            payload = json.loads(self.conversation_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self.conversations = payload
        except Exception:
            self.conversations = []

    def save_conversations(self) -> None:
        self.conversation_file.write_text(
            json.dumps(self.conversations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def new_conversation(self) -> int:
        conv = {
            "id": str(int(time.time() * 1000)),
            "title": f"Chat {len(self.conversations) + 1}",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "turns": [],
        }
        self.conversations.append(conv)
        self.save_conversations()
        return len(self.conversations) - 1

    def delete_conversation(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.conversations):
            return
        self.conversations.pop(idx)
        self.save_conversations()
        if not self.conversations:
            self.new_conversation()
        self.current_conv_index = min(idx, len(self.conversations) - 1)

    # -- model ---------------------------------------------------------------

    def reload_models(self) -> int:
        root = Path(self.config.model_root).expanduser().resolve()
        self.models = discover_models(root)
        return len(self.models)

    def resolve_model(self) -> Any:
        root = Path(self.config.model_root).expanduser().resolve()
        if not self.models:
            self.models = discover_models(root)
        return resolve_model(self.config.model, self.models, root)

    def ensure_model_loaded(self, selected_path: Path) -> None:
        current = str(selected_path.resolve())
        engine_key = f"{self.config.engine}:{self.config.llama_cpp_url}:{self.config.llama_cpp_model}"
        if self.loaded_model is not None and self.loaded_model_path == current and self.loaded_engine_key == engine_key:
            return
        if self.config.engine == "llamacpp":
            model, gen_cfg = load_local_model(
                None,
                engine="llamacpp",
                llama_cpp_url=self.config.llama_cpp_url,
                llama_cpp_model=self.config.llama_cpp_model,
            )
        else:
            model, gen_cfg = load_local_model(selected_path)
        self.loaded_model = model
        self.generation_config_factory = gen_cfg
        self.loaded_model_path = current
        self.loaded_engine_key = engine_key


# ---------------------------------------------------------------------------
# UI Page
# ---------------------------------------------------------------------------

@ui.page("/")
def index_page() -> None:
    state = SessionState()
    state.load_conversations()
    if not state.conversations:
        state.new_conversation()
    state.reload_models()

    # Refs to dynamic UI elements — populated during layout build below
    chat_column: ui.column | None = None
    chat_scroll: ui.scroll_area | None = None
    conv_column: ui.column | None = None
    evidence_label: ui.markdown | None = None
    tools_label: ui.markdown | None = None
    plan_label: ui.markdown | None = None
    metrics_label: ui.label | None = None
    msg_input: ui.input | None = None

    # ---------------------------------------------------------------
    # UI helpers — MUST only be called from the main thread
    # ---------------------------------------------------------------

    def refresh_metrics() -> None:
        if metrics_label is None:
            return
        m = state.metrics
        metrics_label.set_text(
            f"iter={m['iteration']}  |  prompt={m['prompt_tokens']}  |  "
            f"gen={m['generated_tokens']}  |  ttft={float(m['ttft']):.2f}s  |  "
            f"tps={float(m['tps']):.1f}"
        )

    def add_chat_card(kind: str, text: str) -> None:
        if chat_column is None:
            return
        color_map = {
            "user":        "#1b5e20",
            "assistant":   "#006064",
            "tool":        "#f57f17",
            "observation": "#6a1b9a",
            "error":       "#b71c1c",
            "system":      "#37474f",
        }
        bg = color_map.get(kind, "#37474f")
        with chat_column:
            with ui.card().style(
                f"width:100%; border-left:4px solid {bg}; margin-bottom:8px;"
            ):
                ui.label(kind.upper()).style(
                    f"color:{bg}; font-weight:bold; font-size:0.75rem;"
                )
                ui.markdown(text).style("white-space:pre-wrap; font-size:0.85rem;")
        if chat_scroll is not None:
            chat_scroll.scroll_to(percent=1.0)

    def clear_chat() -> None:
        if chat_column is not None:
            chat_column.clear()

    def rebuild_conversation_list() -> None:
        if conv_column is None:
            return
        conv_column.clear()
        with conv_column:
            for idx, conv in enumerate(state.conversations):
                _build_conv_item(idx, conv)

    def _build_conv_item(idx: int, conv: dict[str, Any]) -> None:
        title = conv.get("title", f"Chat {idx + 1}")
        subtitle = (
            f"{_fmt_time(conv.get('updated_at', ''))} · "
            f"{len(conv.get('turns', []))} turns"
        )
        with ui.card().style(
            "width:100%; cursor:pointer; margin-bottom:4px;"
        ).on("click", lambda _e, i=idx: select_conversation(i)):
            with ui.row().style("width:100%; align-items:center;"):
                with ui.column().style("width:100%;"):
                    ui.label(title).style("font-weight:bold; font-size:0.85rem;")
                    ui.label(subtitle).style("font-size:0.7rem; color:#999;")
                ui.button("×", on_click=lambda _e, i=idx: do_delete_conversation(i)).props("flat color=red size=sm")

    def do_delete_conversation(idx: int) -> None:
        state.delete_conversation(idx)
        rebuild_conversation_list()
        select_conversation(state.current_conv_index)

    def select_conversation(idx: int) -> None:
        if idx < 0 or idx >= len(state.conversations):
            return
        state.current_conv_index = idx
        clear_chat()
        for turn in state.conversations[idx].get("turns", []):
            add_chat_card("user", turn.get("user", ""))
            add_chat_card("assistant", turn.get("answer", ""))
        add_chat_card(
            "system",
            f"Opened **{state.conversations[idx].get('title', '')}**",
        )

    def refresh_evidence() -> None:
        if evidence_label is None:
            return
        items = state.dashboard.get("evidences", [])
        if not items:
            evidence_label.set_content("*No evidence yet.*")
            return
        lines = []
        for ev in items[:6]:
            lines.append(
                f"- **{ev.get('value')}**: score={ev.get('score')}  "
                f"usage={ev.get('trusted_usage_hits')}  "
                f"value={ev.get('trusted_value_hits')}"
            )
        evidence_label.set_content("\n".join(lines))

    def refresh_tools() -> None:
        if tools_label is None:
            return
        ts = state.dashboard.get("tool_stats", {})
        counts = ts.get("counts", {})
        if not counts:
            tools_label.set_content("*No tool stats yet.*")
            return
        parts = [f"`{n}`:{counts[n]}" for n in sorted(counts)]
        tools_label.set_content(
            f"calls: {', '.join(parts)}  \nfailures: {ts.get('failures', 0)}"
        )

    def refresh_plan() -> None:
        if plan_label is None:
            return
        p = state.dashboard.get("plan", {})
        if not p:
            plan_label.set_content("*No plan yet.*")
            return
        steps = p.get("steps", [])
        completed = set(p.get("completed_steps", []))
        lines = [
            f"**Goal:** {p.get('goal', '')}",
            f"**Current:** {p.get('current_step', '')}",
            f"**Done when:** {p.get('done_when', '')}",
            "",
        ]
        for step in steps:
            if step in completed:
                lines.append(f"- [x] ~~{step}~~")
            else:
                lines.append(f"- [ ] {step}")
        plan_label.set_content("\n".join(lines))

    # ---------------------------------------------------------------
    # Queue drain — called by ui.timer on the main thread
    # ---------------------------------------------------------------

    def drain_ui_queue() -> None:
        count = 0
        try:
            while True:
                fn = state.ui_queue.get_nowait()
                count += 1
                fn()
        except _queue.Empty:
            pass

    # ---------------------------------------------------------------
    # Event callback — called from worker thread; only enqueues
    # ---------------------------------------------------------------

    def on_event(event: dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        iteration = event.get("iteration", "?")

        if kind == "stream_chunk":
            chunk = str(event.get("chunk", ""))
            if chunk:
                state.stream_buffer += chunk
            return

        if kind == "iteration_start":
            state.metrics["iteration"] = iteration
            state.stream_buffer = ""
            return

        if kind == "prefill_complete":
            state.metrics["prompt_tokens"] = int(event.get("prompt_tokens", 0) or 0)
            state.metrics["ttft"] = float(event.get("first_token_latency", 0.0) or 0.0)
            state.ui_queue.put(refresh_metrics)
            return

        if kind == "decode_stats":
            state.metrics["generated_tokens"] = int(event.get("generated_tokens", 0) or 0)
            state.metrics["tps"] = float(event.get("tps", 0.0) or 0.0)
            state.ui_queue.put(refresh_metrics)
            return

        if kind == "dashboard":
            state.dashboard = {
                "evidences": event.get("evidences", []),
                "tool_stats": event.get("tool_stats", {"counts": {}, "failures": 0}),
                "plan": event.get("plan", {}),
            }
            state.ui_queue.put(refresh_evidence)
            state.ui_queue.put(refresh_tools)
            state.ui_queue.put(refresh_plan)
            return

        if kind == "plan_state":
            state.dashboard["plan"] = event.get("plan", {})
            state.ui_queue.put(refresh_plan)
            return

        if kind == "model_output":
            text = str(event.get("text", ""))
            if text:
                state.stream_buffer += text
            buf = state.stream_buffer.strip()
            if buf:
                preview = buf[-800:]
                state.ui_queue.put(
                    lambda p=preview: add_chat_card("assistant", f"```\n{p}\n```")
                )
                state.stream_buffer = ""
            return

        if kind == "observation":
            decision = event.get("decision", {})
            observation = event.get("observation", {})

            # Flush any buffered stream text
            buf = state.stream_buffer.strip()
            if buf:
                preview = buf[-800:]
                state.ui_queue.put(
                    lambda p=preview: add_chat_card("assistant", f"```\n{p}\n```")
                )
                state.stream_buffer = ""

            # Tool card
            if decision.get("type") == "tool":
                tool_info = json.dumps(
                    {
                        "name": decision.get("name"),
                        "arguments": decision.get("arguments", {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                state.ui_queue.put(
                    lambda t=tool_info: add_chat_card("tool", f"```json\n{t}\n```")
                )

            # Observation card
            ok = observation.get("ok", True)
            focus: dict[str, Any] = {"ok": ok}
            for key in ("error", "hint", "matches", "returncode", "truncated", "total_matches"):
                if key in observation:
                    focus[key] = observation[key]
            if observation.get("output"):
                focus["output_preview"] = str(observation["output"])[:400]
            obs_text = json.dumps(focus, ensure_ascii=False, indent=2)
            card_kind = "error" if not ok else "observation"
            state.ui_queue.put(
                lambda k=card_kind, t=obs_text: add_chat_card(k, f"```json\n{t}\n```")
            )
            return

        if kind == "final":
            answer = str(event.get("answer", ""))
            state.ui_queue.put(
                lambda a=answer: add_chat_card(
                    "assistant", f"### Final Answer\n\n{a}"
                )
            )
            return

    # ---------------------------------------------------------------
    # Run / stop
    # ---------------------------------------------------------------

    def _q(kind: str, text: str) -> None:
        """Enqueue an add_chat_card call (thread-safe)."""
        state.ui_queue.put(lambda k=kind, t=text: add_chat_card(k, t))

    def do_run() -> None:
        if msg_input is None:
            return
        msg = msg_input.value.strip()
        if not msg:
            add_chat_card("system", "Message is empty.")
            return
        if state.running:
            add_chat_card("system", "Run already in progress.")
            return

        msg_input.value = ""
        state.running = True
        state.cancel_requested = False
        state.stream_buffer = ""
        add_chat_card("user", msg)
        add_chat_card("system", "Starting run…")

        def worker() -> None:
            started = time.perf_counter()
            import traceback
            try:
                if state.config.engine == "llamacpp":
                    _q("system", f"Model: **{state.config.llama_cpp_model}**")
                    state.ensure_model_loaded(Path(""))
                else:
                    selected = state.resolve_model()
                    _q("system", f"Model: **{selected.name}**")
                    state.ensure_model_loaded(selected.path)
                if state.config.turboquant:
                    patch_mlx_lm_prompt_cache_with_turboquant(
                        r_bits=state.config.tq_r_bits,
                        theta_bits=state.config.tq_theta_bits,
                    )

                conversation = state.conversations[state.current_conv_index]
                context_turns = conversation.get("turns", [])[-3:]
                context_lines: list[str] = []
                for turn in context_turns:
                    context_lines.append(f"- user: {turn.get('user', '')}")
                    context_lines.append(f"  assistant: {turn.get('answer', '')}")
                context = ""
                if context_lines:
                    context = (
                        "Recent conversation context:\n"
                        + "\n".join(context_lines)
                        + "\n\n"
                    )
                task = context + "Current user request:\n" + msg

                result = run_loop(
                    state.loaded_model,
                    state.generation_config_factory,
                    task=task,
                    workspace=Path(state.config.workspace).expanduser().resolve(),
                    max_iterations=state.config.max_iterations,
                    max_tokens=state.config.max_tokens,
                    temperature=state.config.temperature,
                    stream_output=True,
                    reflection_strength=state.config.reflection_strength,
                    kv_bits=state.config.kv_bits,
                    kv_group_size=state.config.kv_group_size,
                    quantized_kv_start=state.config.quantized_kv_start,
                    event_callback=on_event,
                    should_stop=lambda: state.cancel_requested,
                    compress_observations=state.config.compress_observations,
                    transcript_window=state.config.transcript_window,
                    multi_tool=state.config.multi_tool,
                    use_chat=state.config.use_chat,
                )
                elapsed = round(time.perf_counter() - started, 3)

                if state.cancel_requested or result.answer == "cancelled":
                    _q("system", "Run cancelled.")
                    return

                conversation["turns"].append({
                    "timestamp": _now_iso(),
                    "user": msg,
                    "answer": result.answer,
                    "success": result.success,
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "elapsed_seconds": elapsed,
                    "transcript": result.transcript,
                })
                conversation["updated_at"] = _now_iso()
                state.save_conversations()
                state.metrics["iteration"] = result.iterations
                state.ui_queue.put(rebuild_conversation_list)
                state.ui_queue.put(refresh_metrics)
                _q(
                    "system",
                    f"Run complete: success={result.success}, {elapsed}s, "
                    f"{result.iterations} iterations, {result.tool_calls} tool calls",
                )
            except Exception as exc:
                tb = traceback.format_exc()
                _q("error", f"**Run failed:** {type(exc).__name__}: {exc}\n\n```\n{tb}\n```")
            finally:
                state.running = False

        Thread(target=worker, daemon=True).start()

    def do_stop() -> None:
        if not state.running:
            add_chat_card("system", "No active run.")
            return
        state.cancel_requested = True
        add_chat_card("system", "Stop requested.")

    def do_new_conversation() -> None:
        idx = state.new_conversation()
        rebuild_conversation_list()
        select_conversation(idx)

    # ---------------------------------------------------------------
    # Params dialog
    # ---------------------------------------------------------------

    def open_params_dialog() -> None:
        cfg = state.config

        with ui.dialog() as dlg, ui.card().style("min-width:500px;"):
            ui.label("Parameters").style(
                "font-size:1.2rem; font-weight:bold; margin-bottom:12px;"
            )

            p_model = ui.input("Model", value=cfg.model).style("width:100%;")
            p_model_root = ui.input("Model Root", value=cfg.model_root).style("width:100%;")
            p_workspace = ui.input("Workspace", value=cfg.workspace).style("width:100%;")

            ui.label("Engine").style("font-weight:bold; margin-top:8px;")
            p_engine = ui.radio({"mlx": "MLX", "llamacpp": "llama.cpp"}, value=cfg.engine)

            ui.separator()
            ui.label("=== llama.cpp Parameters ===").style("font-weight:bold; margin-top:8px;")
            p_llama_url = ui.input("LlamaCpp URL", value=cfg.llama_cpp_url).style("width:100%;")
            p_llama_model = ui.input("LlamaCpp Model", value=cfg.llama_cpp_model).style("width:100%;")
            initial_use_chat = True if cfg.engine == "llamacpp" else cfg.use_chat
            ui.label("llama.cpp requires Use Chat=True for structured output").style("font-size:0.75rem; color:#888; margin-bottom:4px;")
            p_use_chat = ui.checkbox("Use Chat", value=initial_use_chat)

            ui.separator()
            with ui.row():
                p_max_iter = ui.number(
                    "Max Iterations", value=cfg.max_iterations, min=1, max=100, step=1
                )
                p_max_tok = ui.number(
                    "Max Tokens", value=cfg.max_tokens, min=64, max=8192, step=64
                )
                p_temp = ui.number(
                    "Temperature",
                    value=cfg.temperature,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    format="%.2f",
                )

            p_reflection = ui.select(
                ["none", "low", "medium", "high"],
                value=cfg.reflection_strength,
                label="Reflection Strength",
            )

            with ui.row():
                p_kv_bits = ui.input(
                    "KV Bits",
                    value="" if cfg.kv_bits is None else str(cfg.kv_bits),
                )
                p_kv_group = ui.input(
                    "KV Group Size",
                    value="" if cfg.kv_group_size is None else str(cfg.kv_group_size),
                )
                p_kv_start = ui.input(
                    "Quantized KV Start",
                    value="" if cfg.quantized_kv_start is None else str(cfg.quantized_kv_start),
                )

            p_tq = ui.checkbox("TurboQuant", value=cfg.turboquant)
            with ui.row():
                p_tq_r = ui.number(
                    "TQ r_bits", value=cfg.tq_r_bits, min=1, max=16, step=1
                )
                p_tq_t = ui.number(
                    "TQ theta_bits", value=cfg.tq_theta_bits, min=1, max=16, step=1
                )

            ui.separator()
            ui.label("[exp] Experimental").style("font-weight:bold; font-size:0.9rem;")
            with ui.row():
                p_compress = ui.checkbox("[exp] Compress Observations", value=cfg.compress_observations)
                p_multi_tool = ui.checkbox("[exp] Multi-Tool", value=cfg.multi_tool)

            p_tw = ui.input(
                "[exp] Transcript Window (blank = off)",
                value="" if cfg.transcript_window is None else str(cfg.transcript_window),
            ).style("width:100%;")

            with ui.row().style("margin-top:16px;"):
                def save() -> None:
                    try:
                        engine = p_engine.value
                        use_chat = p_use_chat.value
                        if engine == "llamacpp":
                            use_chat = True
                        state.config = GuiConfig(
                            model=p_model.value.strip(),
                            model_root=p_model_root.value.strip(),
                            engine=p_engine.value,
                            llama_cpp_url=p_llama_url.value.strip(),
                            llama_cpp_model=p_llama_model.value.strip(),
                            workspace=p_workspace.value.strip(),
                            max_iterations=int(p_max_iter.value),
                            max_tokens=int(p_max_tok.value),
                            temperature=float(p_temp.value),
                            reflection_strength=p_reflection.value,
                            kv_bits=_parse_optional_int(p_kv_bits.value),
                            kv_group_size=_parse_optional_int(p_kv_group.value),
                            quantized_kv_start=_parse_optional_int(p_kv_start.value),
                            turboquant=p_tq.value,
                            tq_r_bits=int(p_tq_r.value),
                            tq_theta_bits=int(p_tq_t.value),
                            use_chat=use_chat,
                            compress_observations=p_compress.value,
                            transcript_window=_parse_optional_int(p_tw.value),
                            multi_tool=p_multi_tool.value,
                        )
                        n = state.reload_models()
                        add_chat_card(
                            "system", f"Parameters updated. {n} models found."
                        )
                    except Exception as exc:
                        ui.notify(f"Invalid params: {exc}", type="negative")
                        return
                    dlg.close()

                ui.button("Save", on_click=save, color="green")
                ui.button("Cancel", on_click=dlg.close)

        dlg.open()

    # ---------------------------------------------------------------
    # Build layout
    # ---------------------------------------------------------------

    ui.dark_mode().enable()

    # Top metrics bar
    with ui.row().style(
        "width:100%; padding:8px 16px; background:#1e1e2e; "
        "border-bottom:1px solid #333; align-items:center; gap:16px;"
    ):
        ui.label("MiniForensicsAgent").style(
            "font-weight:bold; font-size:1.1rem; color:#90caf9;"
        )
        metrics_label = ui.label(
            "iter=0 | prompt=0 | gen=0 | ttft=0.00s | tps=0.0"
        ).style("font-family:monospace; font-size:0.85rem; color:#aaa;")

    # Main three-column layout
    with ui.row().style("width:100%; height:calc(100vh - 50px); gap:0;"):

        # -- Left: conversations --
        with ui.column().style(
            "width:22%; height:100%; padding:12px; "
            "border-right:1px solid #333; overflow-y:auto;"
        ):
            ui.label("Conversations").style("font-weight:bold; margin-bottom:8px;")
            with ui.row().style("gap:4px; margin-bottom:8px;"):
                ui.button("New", on_click=do_new_conversation, color="primary").props(
                    "dense size=sm"
                )
                ui.button("Params", on_click=open_params_dialog).props(
                    "dense size=sm outline"
                )
            conv_column = ui.column().style("width:100%; gap:0;")
            rebuild_conversation_list()

        # -- Middle: chat --
        with ui.column().style("width:52%; height:100%; padding:12px;"):
            ui.label("Agent Chat").style("font-weight:bold; margin-bottom:8px;")
            chat_scroll = ui.scroll_area().style("flex:1; width:100%; height:calc(100% - 80px);")
            with chat_scroll:
                chat_column = ui.column().style("width:100%; gap:0;")
            with ui.row().style("width:100%; margin-top:8px; gap:8px;"):
                msg_input = (
                    ui.input(placeholder="Enter your message…")
                    .style("flex:1;")
                    .on("keydown.enter", lambda _: do_run())
                )
                ui.button("Run", on_click=do_run, color="green").props("dense")
                ui.button("Stop", on_click=do_stop, color="orange").props("dense")

        # -- Right: dashboard --
        with ui.column().style(
            "width:26%; height:100%; padding:12px; "
            "border-left:1px solid #333; overflow-y:auto;"
        ):
            ui.label("Evidences").style("font-weight:bold; margin-bottom:4px;")
            evidence_label = ui.markdown("*No evidence yet.*").style("font-size:0.8rem;")

            ui.separator()
            ui.label("Tool Stats").style("font-weight:bold; margin-bottom:4px;")
            tools_label = ui.markdown("*No tool stats yet.*").style("font-size:0.8rem;")

            ui.separator()
            ui.label("Plan").style("font-weight:bold; margin-bottom:4px;")
            plan_label = ui.markdown("*No plan yet.*").style("font-size:0.8rem;")

    # Timer to drain the thread-safe UI queue (100ms interval)
    ui.timer(0.1, drain_ui_queue)

    # Initial messages
    add_chat_card("system", "Ready. Select a conversation or create a new one.")
    add_chat_card("system", "Tip: configure model and workspace in **Params** before running.")
    select_conversation(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ui.run(title="MiniForensicsAgent", port=8090, reload=False)
    return 0
