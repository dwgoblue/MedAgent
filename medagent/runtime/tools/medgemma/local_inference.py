from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MedGemmaConfig:
    model_ref: str
    use_local_files_only: bool
    device: str
    dtype: str
    max_new_tokens: int
    temperature: float

    @classmethod
    def from_env(cls) -> "MedGemmaConfig":
        model_ref = os.getenv("MEDAGENT_MEDGEMMA_MODEL_DIR", "").strip() or os.getenv(
            "MEDAGENT_MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it"
        )
        return cls(
            model_ref=model_ref,
            use_local_files_only=os.getenv("MEDAGENT_MEDGEMMA_LOCAL_ONLY", "1") == "1",
            device=os.getenv("MEDAGENT_MEDGEMMA_DEVICE", "auto").strip().lower(),
            dtype=os.getenv("MEDAGENT_MEDGEMMA_DTYPE", "auto").strip().lower(),
            max_new_tokens=int(os.getenv("MEDAGENT_MEDGEMMA_MAX_NEW_TOKENS", "384")),
            temperature=float(os.getenv("MEDAGENT_MEDGEMMA_TEMPERATURE", "0.2")),
        )


class LocalMedGemmaClient:
    def __init__(self, config: MedGemmaConfig | None = None) -> None:
        self.config = config or MedGemmaConfig.from_env()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._target_device = "cpu"

    def _resolve_dtype(self, torch_mod: Any) -> Any:
        if self.config.dtype == "bfloat16":
            return torch_mod.bfloat16
        if self.config.dtype == "float16":
            return torch_mod.float16
        if self.config.dtype == "float32":
            return torch_mod.float32
        return torch_mod.bfloat16 if self._target_device == "cuda" else torch_mod.float32

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        if self.config.device in {"cuda", "cpu"}:
            self._target_device = self.config.device
        else:
            self._target_device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = self._resolve_dtype(torch)
        kwargs = {
            "local_files_only": self.config.use_local_files_only,
        }
        self._processor = AutoProcessor.from_pretrained(self.config.model_ref, **kwargs)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", 1)

        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "local_files_only": self.config.use_local_files_only,
        }

        # Prefer explicit placement. Fallback handles environments without accelerate features.
        try:
            model_kwargs["device_map"] = self._target_device
            self._model = AutoModelForImageTextToText.from_pretrained(self.config.model_ref, **model_kwargs)
        except Exception:
            model_kwargs.pop("device_map", None)
            self._model = AutoModelForImageTextToText.from_pretrained(self.config.model_ref, **model_kwargs)
            if self._target_device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")

        self._model.eval()
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            if getattr(self._model.generation_config, "pad_token_id", None) is None:
                self._model.generation_config.pad_token_id = getattr(
                    self._model.generation_config, "eos_token_id", 1
                )

    def _format_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for m in messages:
            formatted.append(
                {
                    "role": m.get("role", "user"),
                    "content": [{"type": "text", "text": m.get("content", "")}],
                }
            )
        return formatted

    @staticmethod
    def _load_image(path: str) -> Any | None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        try:
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                from PIL import Image

                return Image.open(p).convert("RGB")
            if p.suffix.lower() in {".dcm", ".dicom"}:
                import numpy as np
                import pydicom
                from PIL import Image

                ds = pydicom.dcmread(str(p))
                arr = ds.pixel_array.astype("float32")
                arr = arr - arr.min()
                if arr.max() > 0:
                    arr = arr / arr.max()
                arr = (arr * 255).clip(0, 255).astype("uint8")
                if arr.ndim == 2:
                    return Image.fromarray(arr, mode="L").convert("RGB")
                if arr.ndim == 3:
                    if arr.shape[0] in {1, 3} and arr.shape[0] != arr.shape[-1]:
                        arr = np.transpose(arr, (1, 2, 0))
                    return Image.fromarray(arr).convert("RGB")
        except Exception:
            return None
        return None

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int | None = None) -> str:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        max_tokens = max_new_tokens or self.config.max_new_tokens
        prompt = self._processor.apply_chat_template(
            self._format_messages(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=prompt, return_tensors="pt", padding=True)
        if self._target_device == "cuda" and self._torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        pad_id = self._get_pad_token_id()
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                repetition_penalty=1.05,
                pad_token_id=pad_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        output_tokens = generated_ids[0][input_len:]
        return self._processor.decode(output_tokens, skip_special_tokens=True).strip()

    def _get_pad_token_id(self) -> int:
        """Return pad_token_id for generate() to avoid per-call warnings."""
        if self._processor is None:
            return 1
        tok = getattr(self._processor, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            return tok.pad_token_id
        if tok is not None and getattr(tok, "eos_token_id", None) is not None:
            return tok.eos_token_id
        if self._model is not None and hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            return getattr(self._model.generation_config, "pad_token_id", 1) or getattr(
                self._model.generation_config, "eos_token_id", 1
            )
        return 1

    def generate_text(self, prompt: str, system_prompt: str | None = None, max_new_tokens: int | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_new_tokens=max_new_tokens)

    def generate_text_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        images = [img for img in (self._load_image(p) for p in image_paths[:4]) if img is not None]
        if not images:
            return self.generate_text(prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens)

        max_tokens = max_new_tokens or self.config.max_new_tokens
        content: list[dict[str, Any]] = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        prompt_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=prompt_text, images=images, return_tensors="pt", padding=True)
        if self._target_device == "cuda" and self._torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        pad_id = self._get_pad_token_id()
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                repetition_penalty=1.05,
                pad_token_id=pad_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        output_tokens = generated_ids[0][input_len:]
        return self._processor.decode(output_tokens, skip_special_tokens=True).strip()

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        raw = self.generate_text(prompt, system_prompt=system_prompt, max_new_tokens=max_new_tokens)
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def generate_json_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        system_prompt: str | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        raw = self.generate_text_with_images(
            prompt=prompt,
            image_paths=image_paths,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None


_CLIENT: LocalMedGemmaClient | None = None


def get_default_client() -> LocalMedGemmaClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LocalMedGemmaClient()
    return _CLIENT


def medgemma_enabled() -> bool:
    return os.getenv("MEDAGENT_USE_MEDGEMMA", "0") == "1"
