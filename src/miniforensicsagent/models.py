from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_ROOT = Path.home() / ".lmstudio" / "models"
DEFAULT_AGENT_WORKSPACE = Path.cwd() / "agent_workspace"


@dataclass(frozen=True)
class LocalModel:
    name: str
    path: Path


def looks_like_mlx_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_tokenizer = (path / "tokenizer_config.json").is_file() or (path / "tokenizer.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").is_file()
    return has_config and has_tokenizer and has_weights


def discover_models(root: Path) -> list[LocalModel]:
    if not root.exists():
        return []
    models: list[LocalModel] = []
    for candidate in sorted(root.rglob("*")):
        if looks_like_mlx_model(candidate):
            try:
                display = str(candidate.relative_to(root))
            except ValueError:
                display = candidate.name
            models.append(LocalModel(name=display, path=candidate))
    return models


def format_models(models: Iterable[LocalModel]) -> str:
    return "\n".join(f"{index}. {model.name}\n   {model.path}" for index, model in enumerate(models, start=1))


def resolve_model(selector: str, models: list[LocalModel], root: Path) -> LocalModel:
    selector_path = Path(selector).expanduser()
    if selector_path.exists():
        path = selector_path.resolve()
        if not looks_like_mlx_model(path):
            raise SystemExit(f"Path exists but does not look like an MLX model directory: {path}")
        return LocalModel(name=path.name, path=path)
    if selector.isdigit():
        index = int(selector)
        if 1 <= index <= len(models):
            return models[index - 1]
        raise SystemExit(f"Model index {index} is out of range. Use --list-models to see choices.")
    matches = [model for model in models if selector.lower() in model.name.lower()]
    if not matches:
        raise SystemExit(
            f"No model matched {selector!r} under {root}.\nRun with --list-models or pass an absolute model path."
        )
    if len(matches) > 1:
        raise SystemExit(f"Model selector {selector!r} matched multiple models:\n{format_models(matches)}")
    return matches[0]


def patch_mlx_lm_generate_for_effgen() -> None:
    try:
        import importlib
        import inspect

        import mlx_lm
        generate_module = importlib.import_module("mlx_lm.generate")
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
    except Exception:
        return

    if "temp" in inspect.signature(generate_step).parameters:
        return
    if getattr(mlx_lm.generate, "_mini_forensics_agent_compat", False):
        return

    original_generate = mlx_lm.generate
    original_stream_generate = mlx_lm.stream_generate

    def translate_generation_kwargs(kwargs: dict) -> None:
        temp = kwargs.pop("temp", None)
        top_p = kwargs.pop("top_p", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        seed = kwargs.pop("seed", None)

        if seed is not None:
            try:
                import mlx.core as mx

                mx.random.seed(seed)
            except Exception:
                pass

        if "sampler" not in kwargs and (temp is not None or top_p is not None):
            kwargs["sampler"] = make_sampler(
                temp=0.0 if temp is None else temp,
                top_p=0.0 if top_p is None else top_p,
            )

        if "logits_processors" not in kwargs and repetition_penalty not in (None, 1, 1.0):
            kwargs["logits_processors"] = make_logits_processors(repetition_penalty=repetition_penalty)

    def compatible_generate(model, tokenizer, prompt, verbose=False, **kwargs):
        translate_generation_kwargs(kwargs)
        return original_generate(model, tokenizer, prompt, verbose=verbose, **kwargs)

    def compatible_stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None, **kwargs):
        translate_generation_kwargs(kwargs)
        return original_stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, draft_model=draft_model, **kwargs)

    compatible_generate._mini_forensics_agent_compat = True  # type: ignore[attr-defined]
    compatible_stream_generate._mini_forensics_agent_compat = True  # type: ignore[attr-defined]
    mlx_lm.generate = compatible_generate
    mlx_lm.stream_generate = compatible_stream_generate
    generate_module.generate = compatible_generate
    generate_module.stream_generate = compatible_stream_generate


def prepare_effgen_imports():
    patch_mlx_lm_generate_for_effgen()
    try:
        from effgen import GenerationConfig, load_model
    except Exception as exc:
        raise SystemExit(
            "Could not import effGen. Install dependencies with:\n  uv sync\n\n"
            f"Import error: {exc}"
        ) from exc
    return GenerationConfig, load_model


def load_local_model(model_path: Path):
    GenerationConfig, load_model = prepare_effgen_imports()
    return load_model(str(model_path), engine="mlx"), GenerationConfig
