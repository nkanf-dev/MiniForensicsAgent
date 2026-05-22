from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Select, Static
from textual.worker import Worker

from .loop import run_loop
from .models import (
    DEFAULT_AGENT_WORKSPACE,
    DEFAULT_MODEL_ROOT,
    discover_models,
    load_local_model,
    patch_mlx_lm_prompt_cache_with_turboquant,
    resolve_model,
)


@dataclass
class TuiConfig:
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


def _parse_optional_int(raw: str) -> int | None:
    raw = raw.strip()
    if raw == "":
        return None
    return int(raw)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _fmt_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
    except Exception:
        return ts


class ConversationItem(ListItem):
    def __init__(self, conv_index: int, title: str, subtitle: str) -> None:
        super().__init__(Static(f"{title}\n{subtitle}", markup=False))
        self.conv_index = conv_index


class CardWidget(Static):
    def __init__(self, text: str, *, classes: str = "") -> None:
        super().__init__(text, classes=classes, markup=False)
        self.card_text = text
        self.can_focus = True

    def set_card_text(self, text: str) -> None:
        self.card_text = text
        self.update(text)


class ParamsModal(ModalScreen[dict[str, Any] | None]):
    CSS = """
    ParamsModal {
        align: center middle;
    }
    #params-dialog {
        width: 88;
        height: 90%;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }
    #params-scroll {
        height: 1fr;
    }
    .row {
        height: 3;
    }
    .label {
        width: 30;
    }
    """

    def __init__(self, config: TuiConfig) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        with Vertical(id="params-dialog"):
            yield Static("Parameters")
            with VerticalScroll(id="params-scroll"):
                yield self._row("Model", Input(self.config.model, id="p_model"))
                yield self._row("Model Root", Input(self.config.model_root, id="p_model_root"))
                yield self._row("Workspace", Input(self.config.workspace, id="p_workspace"))
                engine_select = Select(
                    id="p_engine",
                    options=[("MLX", "mlx"), ("llama.cpp", "llamacpp")],
                    value=self.config.engine,
                )
                yield self._row("Engine", engine_select)
                yield Static("=== llama.cpp Parameters ===", id="llama-params-label")
                yield self._row("LLamaCpp URL", Input(self.config.llama_cpp_url, id="p_llama_cpp_url"))
                yield self._row("LLamaCpp Model", Input(self.config.llama_cpp_model, id="p_llama_cpp_model"))
                yield self._row("Max Iterations", Input(str(self.config.max_iterations), id="p_max_iterations"))
                yield self._row("Max Tokens", Input(str(self.config.max_tokens), id="p_max_tokens"))
                yield self._row("Temperature", Input(str(self.config.temperature), id="p_temperature"))
                yield self._row("Reflection", Input(self.config.reflection_strength, id="p_reflection"))
                yield self._row("KV Bits", Input("" if self.config.kv_bits is None else str(self.config.kv_bits), id="p_kv_bits"))
                yield self._row("KV Group Size", Input("" if self.config.kv_group_size is None else str(self.config.kv_group_size), id="p_kv_group_size"))
                yield self._row("Quantized KV Start", Input("" if self.config.quantized_kv_start is None else str(self.config.quantized_kv_start), id="p_quantized_kv_start"))
                yield self._row("TurboQuant", Checkbox(label="Enable", value=self.config.turboquant, id="p_turboquant"))
                yield self._row("TQ r_bits", Input(str(self.config.tq_r_bits), id="p_tq_r_bits"))
                yield self._row("TQ theta_bits", Input(str(self.config.tq_theta_bits), id="p_tq_theta_bits"))
                yield self._row("Use Chat", Checkbox(label="Enable", value=self.config.use_chat, id="p_use_chat"))
                yield self._row("[exp] Compress Obs", Checkbox(label="Enable", value=self.config.compress_observations, id="p_compress_obs"))
                yield self._row("[exp] Transcript Window", Input("" if self.config.transcript_window is None else str(self.config.transcript_window), id="p_transcript_window"))
                yield self._row("[exp] Multi-Tool", Checkbox(label="Enable", value=self.config.multi_tool, id="p_multi_tool"))
            with Horizontal():
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")

    def _row(self, label: str, widget: Input | Checkbox) -> Horizontal:
        return Horizontal(Label(label, classes="label"), widget, classes="row")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        try:
            payload = {
                "model": self.query_one("#p_model", Input).value.strip(),
                "model_root": self.query_one("#p_model_root", Input).value.strip(),
                "workspace": self.query_one("#p_workspace", Input).value.strip(),
                "engine": self.query_one("#p_engine", Select).value,
                "llama_cpp_url": self.query_one("#p_llama_cpp_url", Input).value.strip(),
                "llama_cpp_model": self.query_one("#p_llama_cpp_model", Input).value.strip(),
                "max_iterations": int(self.query_one("#p_max_iterations", Input).value.strip()),
                "max_tokens": int(self.query_one("#p_max_tokens", Input).value.strip()),
                "temperature": float(self.query_one("#p_temperature", Input).value.strip()),
                "reflection_strength": self.query_one("#p_reflection", Input).value.strip(),
                "kv_bits": _parse_optional_int(self.query_one("#p_kv_bits", Input).value),
                "kv_group_size": _parse_optional_int(self.query_one("#p_kv_group_size", Input).value),
                "quantized_kv_start": _parse_optional_int(self.query_one("#p_quantized_kv_start", Input).value),
                "turboquant": self.query_one("#p_turboquant", Checkbox).value,
                "tq_r_bits": int(self.query_one("#p_tq_r_bits", Input).value.strip()),
                "tq_theta_bits": int(self.query_one("#p_tq_theta_bits", Input).value.strip()),
                "use_chat": True if self.query_one("#p_engine", Select).value == "llamacpp" else self.query_one("#p_use_chat", Checkbox).value,
                "compress_observations": self.query_one("#p_compress_obs", Checkbox).value,
                "transcript_window": _parse_optional_int(self.query_one("#p_transcript_window", Input).value),
                "multi_tool": self.query_one("#p_multi_tool", Checkbox).value,
            }
        except Exception as exc:
            self.notify(f"Invalid params: {exc}", severity="error")
            return
        self.dismiss(payload)


class ForensicsTuiApp(App):
    CSS = """
    #root { height: 1fr; }
    #top-metrics {
        height: 3;
        border: round $accent;
        padding: 0 1;
        margin: 0 1 1 1;
    }
    #left, #middle, #right {
        border: round $boost;
        margin: 0 1;
        padding: 1;
    }
    #left { width: 26%; }
    #middle { width: 48%; }
    #right { width: 26%; }
    .title { text-style: bold; margin-bottom: 1; }
    #conv-list { height: 1fr; }
    #cards { height: 1fr; border: round $surface; }
    #evidence-box, #tool-box, #plan-box {
        height: 1fr;
        border: round $surface;
        padding: 0 1;
        margin-bottom: 1;
    }
    #input-row { height: 3; margin-top: 1; }
    #msg { width: 1fr; }
    ConversationItem {
        border: round $panel;
        margin-bottom: 1;
        padding: 0 1;
    }
    .card {
        border: round $surface;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    .user { border: round green; }
    .assistant { border: round cyan; }
    .tool { border: round yellow; }
    .observation { border: round magenta; }
    .system { border: round $accent; }
    .error { border: round red; }
    .copy-focus { border: round $accent; }
    """

    BINDINGS = [
        ("r", "run", "Run"),
        ("s", "stop", "Stop"),
        ("n", "new_conversation", "New Chat"),
        ("p", "params", "Params"),
        ("c", "copy_card", "Copy Card"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = TuiConfig()
        self.conversation_file = Path.cwd() / ".mini_forensics_conversations.json"
        self.conversations: list[dict[str, Any]] = []
        self.current_conv_index = 0
        self.models: list[Any] = []
        self.loaded_model_path: str | None = None
        self.loaded_engine_key: str | None = None
        self.loaded_model: Any = None
        self.generation_config_factory: Any = None
        self.active_worker: Worker | None = None
        self.cancel_requested = False
        self.stream_cards: dict[int, Static] = {}
        self.stream_buffers: dict[int, str] = {}
        self.current_run_had_final_event = False
        self.last_card_text = ""
        self.metrics_data: dict[str, Any] = {
            "iteration": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "ttft": 0.0,
            "tps": 0.0,
        }
        self.dashboard_data: dict[str, Any] = {
            "evidences": [],
            "tool_stats": {"counts": {}, "failures": 0},
            "plan": {},
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield Static("iter=0 | prompt=0 | gen=0 | ttft=0.00s | tps=0.0", id="top-metrics")
            with Horizontal():
                with Vertical(id="left"):
                    yield Static("Conversations", classes="title")
                    with Horizontal():
                        yield Button("New", id="new", variant="primary")
                        yield Button("Params", id="params")
                        yield Button("Copy", id="copy")
                        yield Button("Delete", id="delete", variant="error")
                    yield ListView(id="conv-list")
                with Vertical(id="middle"):
                    yield Static("Agent Chat", classes="title")
                    yield VerticalScroll(id="cards")
                    with Horizontal(id="input-row"):
                        yield Input(placeholder="Message...", id="msg")
                        yield Button("Run", id="run", variant="success")
                        yield Button("Stop", id="stop", variant="warning")
                with Vertical(id="right"):
                    yield Static("Evidences", classes="title")
                    yield Static("No evidence yet.", id="evidence-box")
                    yield Static("Tool Stats", classes="title")
                    yield Static("No tool stats yet.", id="tool-box")
                    yield Static("Plan", classes="title")
                    yield Static("No plan yet.", id="plan-box")

    def on_mount(self) -> None:
        self._load_conversations()
        if not self.conversations:
            self._new_conversation_internal()
        self._refresh_conversation_list()
        self._select_conversation(0)
        self._reload_models()
        self._refresh_top_metrics()
        self._refresh_right_panels()
        self._mount_status_card("system", "Ready.")
        self._mount_status_card("system", "Tip: focus a card and press c to copy text.")

    def _reload_models(self) -> None:
        root = Path(self.config.model_root).expanduser().resolve()
        self.models = discover_models(root)
        self._mount_status_card("system", f"Models loaded: {len(self.models)}")

    def _load_conversations(self) -> None:
        self.conversations = []
        if not self.conversation_file.exists():
            return
        try:
            payload = json.loads(self.conversation_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self.conversations = payload
        except Exception:
            self.conversations = []

    def _save_conversations(self) -> None:
        self.conversation_file.write_text(json.dumps(self.conversations, ensure_ascii=False, indent=2), encoding="utf-8")

    def _new_conversation_internal(self) -> int:
        conv = {
            "id": str(int(time.time() * 1000)),
            "title": f"Chat {len(self.conversations) + 1}",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "turns": [],
        }
        self.conversations.append(conv)
        self._save_conversations()
        return len(self.conversations) - 1

    def _delete_conversation(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.conversations):
            return
        if len(self.conversations) == 1:
            self.notify("Cannot delete the last conversation", severity="warning")
            return
        self.conversations.pop(idx)
        self._save_conversations()
        self.current_conv_index = min(idx, len(self.conversations) - 1)
        self._refresh_conversation_list()
        self._select_conversation(self.current_conv_index)

    def _refresh_conversation_list(self) -> None:
        view = self.query_one("#conv-list", ListView)
        view.clear()
        for idx, conv in enumerate(self.conversations):
            subtitle = f"{_fmt_time(conv.get('updated_at', ''))} • {len(conv.get('turns', []))} turns"
            view.append(ConversationItem(idx, conv.get("title", f"Chat {idx + 1}"), subtitle))

    def _select_conversation(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.conversations):
            return
        self.current_conv_index = idx
        self.stream_cards = {}
        self.stream_buffers = {}
        cards = self.query_one("#cards", VerticalScroll)
        cards.remove_children()
        for turn in self.conversations[idx].get("turns", []):
            self._mount_card("user", f"User\n{turn.get('user', '')}")
            self._mount_card("assistant", f"Assistant\n{turn.get('answer', '')}")
        self._mount_status_card("system", f"Opened {self.conversations[idx].get('title', '')}")

    def _mount_card(self, kind: str, text: str) -> Static:
        widget = CardWidget(text, classes=f"card {kind}")
        self.query_one("#cards", VerticalScroll).mount(widget)
        self.query_one("#cards", VerticalScroll).scroll_end(animate=False)
        self.last_card_text = text
        return widget

    def _mount_status_card(self, kind: str, text: str) -> None:
        self._mount_card(kind, f"Status\n{text}")

    def _refresh_top_metrics(self) -> None:
        self.query_one("#top-metrics", Static).update(
            "iter={iteration} | prompt={prompt_tokens} | gen={generated_tokens} | ttft={ttft:.2f}s | tps={tps:.1f}".format(
                iteration=self.metrics_data.get("iteration", 0),
                prompt_tokens=self.metrics_data.get("prompt_tokens", 0),
                generated_tokens=self.metrics_data.get("generated_tokens", 0),
                ttft=float(self.metrics_data.get("ttft", 0.0)),
                tps=float(self.metrics_data.get("tps", 0.0)),
            )
        )

    def _refresh_right_panels(self) -> None:
        evidences = self.dashboard_data.get("evidences", [])
        if evidences:
            lines = []
            for item in evidences[:6]:
                lines.append(
                    f"- {item.get('value')}: score={item.get('score')} usage={item.get('trusted_usage_hits')} value={item.get('trusted_value_hits')}"
                )
            self.query_one("#evidence-box", Static).update("\n".join(lines))
        else:
            self.query_one("#evidence-box", Static).update("No evidence yet.")

        tool_stats = self.dashboard_data.get("tool_stats", {})
        counts = tool_stats.get("counts", {})
        if counts:
            parts = [f"{name}:{counts[name]}" for name in sorted(counts.keys())]
            self.query_one("#tool-box", Static).update(
                "calls=" + ", ".join(parts) + f"\nfailures={tool_stats.get('failures', 0)}"
            )
        else:
            self.query_one("#tool-box", Static).update("No tool stats yet.")

        plan = self.dashboard_data.get("plan", {})
        if plan:
            steps = plan.get("steps", [])
            completed = set(plan.get("completed_steps", []))
            lines = [f"goal: {plan.get('goal', '')}", f"current: {plan.get('current_step', '')}", f"done_when: {plan.get('done_when', '')}"]
            for step in steps:
                marker = "[x]" if step in completed else "[ ]"
                lines.append(f"{marker} {step}")
            self.query_one("#plan-box", Static).update("\n".join(lines))
        else:
            self.query_one("#plan-box", Static).update("No plan yet.")

    def _resolve_model(self) -> Any:
        root = Path(self.config.model_root).expanduser().resolve()
        if not self.models:
            self.models = discover_models(root)
        return resolve_model(self.config.model, self.models, root)

    def _ensure_model_loaded(self, selected_path: Path) -> None:
        current = str(selected_path.resolve())
        engine_key = f"{self.config.engine}:{self.config.llama_cpp_url}:{self.config.llama_cpp_model}"
        if self.loaded_model is not None and self.loaded_model_path == current and self.loaded_engine_key == engine_key:
            return
        if self.config.engine == "llamacpp":
            model, generation_config_factory = load_local_model(
                None,
                engine="llamacpp",
                llama_cpp_url=self.config.llama_cpp_url,
                llama_cpp_model=self.config.llama_cpp_model,
            )
        else:
            model, generation_config_factory = load_local_model(selected_path)
        self.loaded_model = model
        self.generation_config_factory = generation_config_factory
        self.loaded_model_path = current
        self.loaded_engine_key = engine_key

    @on(Button.Pressed, "#new")
    def _new_button(self) -> None:
        self.action_new_conversation()

    @on(Button.Pressed, "#params")
    def _params_button(self) -> None:
        self.action_params()

    @on(Button.Pressed, "#copy")
    def _copy_button(self) -> None:
        self.action_copy_card()

    @on(Button.Pressed, "#delete")
    def _delete_button(self) -> None:
        self._delete_conversation(self.current_conv_index)

    @on(Button.Pressed, "#run")
    def _run_button(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#stop")
    def _stop_button(self) -> None:
        self.action_stop()

    @on(Input.Submitted, "#msg")
    def _msg_submit(self) -> None:
        self.action_run()

    @on(ListView.Selected, "#conv-list")
    def _select_from_list(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ConversationItem):
            self._select_conversation(event.item.conv_index)

    @on(events.Click, ".card")
    def _card_clicked(self, event: events.Click) -> None:
        for card in self.query("#cards .card"):
            card.remove_class("copy-focus")
        if event.widget is not None:
            event.widget.add_class("copy-focus")
            if isinstance(event.widget, CardWidget):
                self.last_card_text = event.widget.card_text

    def action_new_conversation(self) -> None:
        idx = self._new_conversation_internal()
        self._refresh_conversation_list()
        self._select_conversation(idx)

    def action_params(self) -> None:
        def apply(result: dict[str, Any] | None) -> None:
            if result is None:
                return
            self.config = TuiConfig(**result)
            self._reload_models()
            self._mount_status_card("system", "Parameters updated.")

        self.push_screen(ParamsModal(self.config), apply)

    def _copy_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        try:
            import pyperclip  # type: ignore

            pyperclip.copy(text)
            return True
        except Exception:
            pass
        try:
            if shutil.which("pbcopy"):
                subprocess.run(["pbcopy"], input=text, text=True, check=False)
                return True
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text, text=True, check=False)
                return True
            if shutil.which("xclip"):
                subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=False)
                return True
        except Exception:
            return False
        return False

    def action_copy_card(self) -> None:
        focused = self.focused
        text = ""
        if isinstance(focused, CardWidget):
            text = focused.card_text
        if not text:
            text = self.last_card_text
        if not text:
            self._mount_status_card("system", "Nothing to copy.")
            return
        ok = self._copy_to_clipboard(text)
        if ok:
            self._mount_status_card("system", "Copied card text to clipboard.")
        else:
            self._mount_status_card("error", "Clipboard tool unavailable. Install pyperclip or use pbcopy.")

    def action_stop(self) -> None:
        if not self.active_worker or not self.active_worker.is_running:
            self._mount_status_card("system", "No active run.")
            return
        self.cancel_requested = True
        self.active_worker.cancel()
        self._mount_status_card("system", "Stop requested.")

    def action_run(self) -> None:
        if self.active_worker and self.active_worker.is_running:
            self._mount_status_card("system", "Run already in progress.")
            return
        msg = self.query_one("#msg", Input).value.strip()
        if not msg:
            self._mount_status_card("system", "Message is empty.")
            return
        self.query_one("#msg", Input).value = ""
        self.cancel_requested = False
        self.stream_cards = {}
        self.stream_buffers = {}
        self.current_run_had_final_event = False
        self._mount_card("user", f"User\n{msg}")
        self.active_worker = self.run_worker(lambda: self._run_impl(msg), thread=True, exclusive=True)

    def _update_stream(self, iteration: int, chunk: str) -> None:
        if iteration not in self.stream_cards:
            self.stream_buffers[iteration] = ""
            self.stream_cards[iteration] = self._mount_card("assistant", f"Model Output (iter {iteration})\n")
        self.stream_buffers[iteration] = self.stream_buffers.get(iteration, "") + chunk
        preview = self.stream_buffers[iteration][-600:]
        text = f"Model Output (iter {iteration})\n{preview}"
        if isinstance(self.stream_cards[iteration], CardWidget):
            self.stream_cards[iteration].set_card_text(text)
        else:
            self.stream_cards[iteration].update(text)

    def _format_tool_card(self, decision: dict[str, Any]) -> str:
        decision_type = str(decision.get("type", ""))
        if decision_type == "tool":
            payload = {
                "name": decision.get("name"),
                "arguments": decision.get("arguments", {}),
            }
            return "Tool Call\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        return "Decision\n" + json.dumps(decision, ensure_ascii=False, indent=2)

    def _format_observation_card(self, observation: dict[str, Any]) -> str:
        focus: dict[str, Any] = {"ok": observation.get("ok", True)}
        for key in (
            "error",
            "hint",
            "matches",
            "read_suggestions",
            "switch_file_suggestions",
            "suggested_read",
            "returncode",
            "truncated",
            "total_matches",
            "omitted_matches",
            "output_path",
            "trimmed_line_texts",
        ):
            if key in observation:
                focus[key] = observation[key]
        if "output" in observation and observation.get("output"):
            output = str(observation.get("output", ""))
            focus["output_preview"] = output[:500]
        return "Observation\n" + json.dumps(focus, ensure_ascii=False, indent=2)

    def _event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type", ""))
        iteration = event.get("iteration", "?")
        if kind == "stream_chunk":
            chunk = str(event.get("chunk", ""))
            if chunk:
                try:
                    iter_no = int(iteration)
                except Exception:
                    iter_no = -1
                self.call_from_thread(self._update_stream, iter_no, chunk)
            return
        if kind == "iteration_start":
            self.metrics_data["iteration"] = int(iteration) if isinstance(iteration, int) else self.metrics_data.get("iteration", 0)
            self.call_from_thread(self._refresh_top_metrics)
            self.call_from_thread(self._mount_status_card, "system", f"Iteration {iteration} started")
            return
        if kind == "prefill_complete":
            self.metrics_data["prompt_tokens"] = int(event.get("prompt_tokens", 0) or 0)
            self.metrics_data["ttft"] = float(event.get("first_token_latency", 0.0) or 0.0)
            self.call_from_thread(self._refresh_top_metrics)
            self.call_from_thread(
                self._mount_status_card,
                "system",
                f"Iter {iteration}: prefill tokens={event.get('prompt_tokens', '?')} first_token={event.get('first_token_latency', 0):.2f}s",
            )
            return
        if kind == "decode_stats":
            self.metrics_data["generated_tokens"] = int(event.get("generated_tokens", 0) or 0)
            self.metrics_data["tps"] = float(event.get("tps", 0.0) or 0.0)
            self.call_from_thread(self._refresh_top_metrics)
            return
        if kind == "dashboard":
            self.dashboard_data = {
                "evidences": event.get("evidences", []),
                "tool_stats": event.get("tool_stats", {"counts": {}, "failures": 0}),
                "plan": event.get("plan", {}),
            }
            self.call_from_thread(self._refresh_right_panels)
            return
        if kind == "plan_state":
            self.dashboard_data["plan"] = event.get("plan", {})
            self.call_from_thread(self._refresh_right_panels)
            return
        if kind == "observation":
            decision = event.get("decision", {})
            observation = event.get("observation", {})
            card_kind = "observation" if observation.get("ok", True) else "error"
            self.call_from_thread(self._mount_card, "tool", self._format_tool_card(decision))
            self.call_from_thread(
                self._mount_card,
                card_kind,
                self._format_observation_card(observation),
            )
            if isinstance(iteration, int) and iteration in self.stream_cards:
                self.call_from_thread(self.stream_cards[iteration].add_class, "system")
            return
        if kind == "final":
            answer = str(event.get("answer", ""))
            self.current_run_had_final_event = True
            self.call_from_thread(self._mount_card, "assistant", f"Assistant (final)\n{answer}")

    def _run_impl(self, msg: str) -> None:
        started = time.perf_counter()
        try:
            if self.config.engine == "llamacpp":
                selected_name = self.config.llama_cpp_model
                self.call_from_thread(self._mount_status_card, "system", f"Model: {selected_name}")
                self._ensure_model_loaded(Path(""))
            else:
                selected = self._resolve_model()
                selected_name = selected.name
                self.call_from_thread(self._mount_status_card, "system", f"Model: {selected_name}")
                self._ensure_model_loaded(selected.path)
            if self.config.turboquant:
                patch_mlx_lm_prompt_cache_with_turboquant(r_bits=self.config.tq_r_bits, theta_bits=self.config.tq_theta_bits)
                self.call_from_thread(self._mount_status_card, "system", "TurboQuant enabled")

            conversation = self.conversations[self.current_conv_index]
            context_turns = conversation.get("turns", [])[-3:]
            context_lines: list[str] = []
            for turn in context_turns:
                context_lines.append(f"- user: {turn.get('user', '')}")
                context_lines.append(f"  assistant: {turn.get('answer', '')}")
            context = ""
            if context_lines:
                context = "Recent conversation context:\n" + "\n".join(context_lines) + "\n\n"
            task = context + "Current user request:\n" + msg

            result = run_loop(
                self.loaded_model,
                self.generation_config_factory,
                task=task,
                workspace=Path(self.config.workspace).expanduser().resolve(),
                max_iterations=self.config.max_iterations,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream_output=True,
                reflection_strength=self.config.reflection_strength,
                kv_bits=self.config.kv_bits,
                kv_group_size=self.config.kv_group_size,
                quantized_kv_start=self.config.quantized_kv_start,
                event_callback=self._event,
                should_stop=lambda: self.cancel_requested,
                compress_observations=self.config.compress_observations,
                transcript_window=self.config.transcript_window,
                multi_tool=self.config.multi_tool,
                use_chat=self.config.use_chat,
            )
            elapsed = round(time.perf_counter() - started, 3)
            if self.cancel_requested or result.answer == "cancelled":
                self.call_from_thread(self._mount_status_card, "system", "Run cancelled.")
                return

            conversation["turns"].append(
                {
                    "timestamp": _now_iso(),
                    "user": msg,
                    "answer": result.answer,
                    "success": result.success,
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "elapsed_seconds": elapsed,
                    "transcript": result.transcript,
                }
            )
            conversation["updated_at"] = _now_iso()
            self._save_conversations()
            self.call_from_thread(self._refresh_conversation_list)
            self.call_from_thread(self._mount_status_card, "system", f"Run complete: success={result.success}, {elapsed}s")
            self.metrics_data["iteration"] = result.iterations
            self.call_from_thread(self._refresh_top_metrics)
            if not self.current_run_had_final_event:
                self.call_from_thread(self._mount_card, "assistant", f"Assistant (summary)\n{result.answer}")
        except Exception as exc:
            self.call_from_thread(self._mount_status_card, "error", f"Run failed: {type(exc).__name__}: {exc}")


def main() -> int:
    app = ForensicsTuiApp()
    app.run()
    return 0
