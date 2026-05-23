from __future__ import annotations

import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    for config_file in root.rglob("config.json"):
        candidate = config_file.parent
        if looks_like_mlx_model(candidate):
            try:
                display = str(candidate.relative_to(root))
            except ValueError:
                display = candidate.name
            models.append(LocalModel(name=display, path=candidate))
    return sorted(models, key=lambda m: m.name)


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


def patch_effgen_mlx_engine_for_kv_cache() -> None:
    try:
        from effgen.models.mlx_engine import MLXEngine
        from effgen.models.base import GenerationConfig
        from mlx_lm import generate as mlx_generate
        from mlx_lm import stream_generate as mlx_stream_generate
    except Exception:
        return

    if getattr(MLXEngine, "_mini_forensics_agent_kv_patch", False):
        return

    def _config_kv_kwargs(config: GenerationConfig) -> dict:
        kv_kwargs: dict = {}
        for name in ("kv_bits", "kv_group_size", "quantized_kv_start"):
            value = getattr(config, name, None)
            if value is not None:
                kv_kwargs[name] = value
        return kv_kwargs

    def generate(self, prompt: str, config: GenerationConfig | None = None, system_prompt: str | None = None, skip_chat_template: bool = False, **kwargs):
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")
        if config is None:
            config = GenerationConfig()
        formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt) if not skip_chat_template else prompt
        self.validate_prompt(formatted_prompt)
        gen_kwargs: dict[str, object] = {
            "max_tokens": config.max_tokens or 512,
            "temp": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            **_config_kv_kwargs(config),
            **kwargs,
        }
        if config.seed is not None:
            gen_kwargs["seed"] = config.seed
        generated_text = mlx_generate(self.model, self.tokenizer, prompt=formatted_prompt, verbose=False, **gen_kwargs)
        if config.stop_sequences:
            for stop_seq in config.stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[: generated_text.index(stop_seq)]
                    break
        prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
        completion_tokens = len(self.tokenizer.encode(generated_text))
        from effgen.models.base import GenerationResult

        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "chat_template_applied": not skip_chat_template and self.apply_chat_template,
            "engine": "mlx",
        }
        metadata.update(_config_kv_kwargs(config))
        return GenerationResult(
            text=generated_text,
            tokens_used=completion_tokens,
            finish_reason="stop",
            model_name=self.model_name,
            metadata=metadata,
        )

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None, system_prompt: str | None = None, skip_chat_template: bool = False, **kwargs):
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")
        if config is None:
            config = GenerationConfig()
        formatted_prompt = self._format_prompt_with_chat_template(prompt, system_prompt) if not skip_chat_template else prompt
        self.validate_prompt(formatted_prompt)
        gen_kwargs: dict[str, object] = {
            "max_tokens": config.max_tokens or 512,
            "temp": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            **_config_kv_kwargs(config),
            **kwargs,
        }
        if config.seed is not None:
            gen_kwargs["seed"] = config.seed
        for response in mlx_stream_generate(self.model, self.tokenizer, prompt=formatted_prompt, **gen_kwargs):
            if isinstance(response, dict):
                text = response.get("text", "")
            elif hasattr(response, "text"):
                text = response.text
            else:
                text = str(response)
            if text:
                if config.stop_sequences:
                    for stop_seq in config.stop_sequences:
                        if stop_seq in text:
                            yield text[: text.index(stop_seq)]
                            return
                yield text

    MLXEngine.generate = generate  # type: ignore[assignment]
    MLXEngine.generate_stream = generate_stream  # type: ignore[assignment]
    MLXEngine._mini_forensics_agent_kv_patch = True  # type: ignore[attr-defined]


def patch_mlx_lm_prompt_cache_with_turboquant(*, r_bits: int = 4, theta_bits: int = 4) -> None:
    """
    Optional backend: TurboQuant KV cache compression.

    This follows turboquant-mlx's recommended integration by monkey-patching
    mlx_lm.models.cache.make_prompt_cache.
    """
    TurboQuantKVCache = None
    import_error: Exception | None = None
    for mod, attr in (
        ("turboquant_mlx.mlx_kvcache", "TurboQuantKVCache"),
        ("turboquant_mlx.mlx_kv_cache", "TurboQuantKVCache"),
        ("turboquant_mlx", "TurboQuantKVCache"),
    ):
        try:
            module = __import__(mod, fromlist=[attr])
            TurboQuantKVCache = getattr(module, attr)
            break
        except Exception as exc:  # pragma: no cover - depends on installed turboquant layout
            import_error = exc

    if TurboQuantKVCache is None:
        raise RuntimeError(
            "TurboQuant backend requested, but TurboQuant KV cache class could not be imported.\n"
            "Install optional deps with:\n"
            "  uv sync --extra turboquant\n"
            "or\n"
            "  uv pip install -e '.[turboquant]'\n"
            "If turboquant is installed but still failing, you're likely on a minimal PyPI build that "
            "does not ship the mlx-lm cache integration. Error: "
            f"{type(import_error).__name__ if import_error else 'Unknown'}: {import_error}"
        )

    import mlx_lm.models.cache as cache_module

    if getattr(cache_module, "_mini_forensics_agent_turboquant_patch", False):
        return

    def turboquant_make_prompt_cache(model, max_kv_size=None):  # noqa: ARG001
        num_layers = len(getattr(model, "layers", []))
        return [TurboQuantKVCache(r_bits=r_bits, theta_bits=theta_bits) for _ in range(num_layers)]

    cache_module.make_prompt_cache = turboquant_make_prompt_cache  # type: ignore[assignment]
    cache_module._mini_forensics_agent_turboquant_patch = True  # type: ignore[attr-defined]


def prepare_effgen_imports():
    patch_mlx_lm_generate_for_effgen()
    patch_effgen_mlx_engine_for_kv_cache()
    try:
        from effgen import GenerationConfig, load_model
    except Exception as exc:
        raise SystemExit(
            "Could not import effGen. Install dependencies with:\n  uv sync\n\n"
            f"Import error: {exc}"
        ) from exc
    return GenerationConfig, load_model


def load_llama_cpp_model(
    base_url: str = "http://localhost:8080/v1",
    model: str = "qwen3.5-9b-instruct.Q4_K_M_deepseek4.gguf",
) -> tuple[Any, type]:
    from .llamacpp import LlamaCPPHTTPClient

    client = LlamaCPPHTTPClient(base_url, model)
    return client, _make_dummy_generation_config()


def _make_dummy_generation_config() -> type:
    class DummyGenerationConfig:
        def __init__(
            self,
            temperature: float = 0.3,
            max_tokens: int = 768,
            top_p: float = 0.9,
            stop_sequences: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p = top_p
            self.stop_sequences = stop_sequences
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __call__(self, **kwargs: Any) -> DummyGenerationConfig:
            return DummyGenerationConfig(**kwargs)

    return DummyGenerationConfig


def discover_llama_cpp_models(base_url: str) -> list[LocalModel]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models: list[LocalModel] = []
        for obj in data.get("data", []):
            mid = obj.get("id", "")
            if mid:
                models.append(LocalModel(name=mid, path=Path(mid)))
        return models
    except Exception:
        return []


def load_local_model(
    model_path: Path | None,
    engine: str = "mlx",
    *,
    llama_cpp_url: str | None = None,
    llama_cpp_model: str | None = None,
    use_chat: bool = False,
):
    if engine == "llamacpp":
        return load_llama_cpp_model(
            base_url=llama_cpp_url or "http://localhost:8080/v1",
            model=llama_cpp_model or "qwen3.5-9b-instruct.Q4_K_M_deepseek4.gguf",
        )
    GenerationConfig, load_model = prepare_effgen_imports()
    return load_model(str(model_path), engine="mlx"), GenerationConfig
