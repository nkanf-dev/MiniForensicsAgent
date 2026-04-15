from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .loop import run_loop
from .models import (
    DEFAULT_AGENT_WORKSPACE,
    DEFAULT_MODEL_ROOT,
    discover_models,
    load_local_model,
    patch_mlx_lm_prompt_cache_with_turboquant,
    resolve_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MiniForensicsAgent.")
    parser.add_argument("--model", default="LocoOperator")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--workspace", default=str(DEFAULT_AGENT_WORKSPACE))
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--reflection-strength", choices=["none", "low", "medium", "high"], default="medium")
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=None)
    parser.add_argument("--quantized-kv-start", type=int, default=None)
    parser.add_argument("--turboquant", action="store_true")
    parser.add_argument("--tq-r-bits", type=int, default=4)
    parser.add_argument("--tq-theta-bits", type=int, default=4)
    parser.add_argument("--task", required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    # Experimental features
    parser.add_argument("--compress-observations", action="store_true", help="[exp] Compress old observation payloads to reduce prompt size (O(n²) → O(n) tokens).")
    parser.add_argument("--transcript-window", type=int, default=None, metavar="K", help="[exp] Only include the last K turns in each prompt (sliding window).")
    parser.add_argument("--multi-tool", action="store_true", help="[exp] Allow multiple independent tool calls per turn.")
    args = parser.parse_args()

    root = Path(args.model_root).expanduser().resolve()
    models = discover_models(root)
    if args.list_models:
        for model in models:
            print(f"{model.name}\t{model.path}")
        return 0
    selected = resolve_model(args.model, models, root)
    workspace = Path(args.workspace).expanduser().resolve()
    started = time.perf_counter()
    model, generation_config = load_local_model(selected.path)
    if args.turboquant:
        patch_mlx_lm_prompt_cache_with_turboquant(r_bits=args.tq_r_bits, theta_bits=args.tq_theta_bits)
    result = run_loop(
        model,
        generation_config,
        task=args.task,
        workspace=workspace,
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream_output=args.stream,
        reflection_strength=args.reflection_strength,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        compress_observations=args.compress_observations,
        transcript_window=args.transcript_window,
        multi_tool=args.multi_tool,
    )
    payload = {
        "model": selected.name,
        "model_path": str(selected.path),
        "success": result.success,
        "answer": result.answer,
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "workspace": str(workspace),
        "kv_cache_quantization": {
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "quantized_kv_start": args.quantized_kv_start,
        },
        "turboquant_kv_cache": {
            "enabled": bool(args.turboquant),
            "r_bits": args.tq_r_bits,
            "theta_bits": args.tq_theta_bits,
        },
        "transcript": result.transcript,
    }
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1
