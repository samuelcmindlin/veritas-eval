"""Model runner: generation, teacher-forced re-scoring, residual-stream
extraction, and steering hooks (ARCHITECTURE `Subject` + `Hook` realization).

Design notes (Stage 0b):
- Raw HF forward hooks on decoder layers, not TransformerLens — a deliberate
  B7-style tool annotation: fewer deps and no weight-folding surprises; hook
  correctness is certified by the two §8 known-answer tests (same-site nulling,
  above-read-layer bit-identity) rather than by framework trust. Swappable
  behind this interface if Stage C prefers TransformerLens/nnsight.
- "Residual stream at layer L" := the hidden state ENTERING decoder layer L
  (resid_pre). Steering at layer k adds a vector to the stream entering k, so
  it affects layers >= k; a probe reading layer L sees an injection at k iff
  k <= L (additive carry), which is what the injection-site assertions and
  known-answer tests rely on.
- Activations STREAM to disk per item (invariant #4): nothing accumulates in
  RAM beyond one forward pass.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

EXTRACTION_CODE_VERSION = "0b-1"  # part of every cache key: bump on hook changes


@dataclass
class SteeringHook:
    """Add `vector` to the residual stream entering `layer` (all positions)."""

    layer: int
    vector: np.ndarray  # (d_model,)
    coefficient: float = 1.0


@dataclass(frozen=True)
class Capture:
    output_text: str
    activations: Mapping[int, np.ndarray] | None  # layer -> (seq, d_model), resid_pre
    meta: Mapping[str, Any] = field(default_factory=dict)


class HFSubject:
    """Hugging Face causal-LM subject with resid_pre extraction + steering."""

    def __init__(self, model_name: str, revision: str, device: str | None = None,
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self._layers = self._decoder_layers()
        self.id = f"{model_name}@{revision}|extract:{EXTRACTION_CODE_VERSION}"

    # -- internals ---------------------------------------------------------

    def _decoder_layers(self) -> list[torch.nn.Module]:
        m = self.model
        for attr in ("model", "transformer"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                for layers_attr in ("layers", "h"):
                    if hasattr(inner, layers_attr):
                        return list(getattr(inner, layers_attr))
        raise ValueError(f"cannot locate decoder layers on {type(m).__name__}")

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def _install(self, read_layers: tuple[int, ...], steering: list[SteeringHook]):
        """Install pre-forward hooks. Returns (handles, store) where store maps
        read layer -> hidden state ENTERING that layer (post-steering if a
        steering hook targets the same layer — matching the additive-carry
        semantics: an injection at k is part of the stream from k onward)."""
        store: dict[int, torch.Tensor] = {}
        steer_by_layer: dict[int, list[SteeringHook]] = {}
        for s in steering:
            steer_by_layer.setdefault(s.layer, []).append(s)
        handles = []

        def make_hook(idx: int):
            def pre_hook(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                for s in steer_by_layer.get(idx, ()):
                    vec = torch.as_tensor(
                        s.vector, dtype=hidden.dtype, device=hidden.device
                    )
                    hidden = hidden + s.coefficient * vec
                if idx in read_layers:
                    store[idx] = hidden.detach()[0].to("cpu", torch.float32)
                if args:
                    return (hidden,) + args[1:], kwargs
                kwargs["hidden_states"] = hidden
                return args, kwargs

            return pre_hook

        touched = set(read_layers) | set(steer_by_layer)
        bad = [i for i in touched if not (0 <= i < len(self._layers))]
        if bad:
            raise ValueError(
                f"layer index(es) {bad} out of range for {len(self._layers)}-layer model"
            )
        for idx in sorted(touched):
            handles.append(
                self._layers[idx].register_forward_pre_hook(make_hook(idx), with_kwargs=True)
            )
        return handles, store

    # -- public surface ----------------------------------------------------

    def generate(self, prompt: str, *, seed: int, max_new_tokens: int = 64,
                 temperature: float = 1.0, do_sample: bool = True) -> Capture:
        """Generate with logged decoding params + seed (DEV §9 item 2)."""
        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return Capture(
            output_text=text,
            activations=None,
            meta={
                "decoding": {
                    "seed": seed, "max_new_tokens": max_new_tokens,
                    "temperature": temperature, "do_sample": do_sample,
                },
                "subject_id": self.id,
            },
        )

    def rescore(self, text: str, *, read_layers: tuple[int, ...],
                steering: list[SteeringHook] | None = None) -> Capture:
        """Teacher-forced forward pass over FIXED tokens, capturing resid_pre
        at `read_layers` and applying steering hooks (IV-B2 / diagnostics path,
        ARCHITECTURE Subject.rescore)."""
        handles, store = self._install(tuple(read_layers), steering or [])
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in handles:
                h.remove()
        return Capture(
            output_text=text,
            activations={k: v.numpy() for k, v in store.items()},
            meta={"subject_id": self.id, "read_layers": list(read_layers),
                  "steered": bool(steering)},
        )


# -- streaming activation cache (invariant #4: never hold the set in RAM) ----

class ActivationCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, subject_id: str, item_id: str, condition: str,
            layers: tuple[int, ...]) -> str:
        payload = json.dumps(
            {"subject": subject_id, "item": item_id, "cond": condition,
             "layers": list(layers)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> dict[int, np.ndarray] | None:
        p = self.path(key)
        if not p.exists():
            return None
        with np.load(p) as z:
            return {int(k): z[k] for k in z.files}

    def put(self, key: str, acts: Mapping[int, np.ndarray]) -> None:
        np.savez_compressed(self.path(key), **{str(k): v for k, v in acts.items()})


def load_subject(model_name: str, revision: str, **kw) -> HFSubject:
    return HFSubject(model_name, revision, **kw)
