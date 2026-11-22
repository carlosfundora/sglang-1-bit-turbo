from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.request


@dataclass(frozen=True)
class ModelKVShape:
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int


class ModelShapeRegistry:
    """Model metadata registry used by the universal broker materialization path."""

    def __init__(self) -> None:
        self._registry: dict[str, ModelKVShape] = {}

    def register(
        self, model_id: str, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        self._registry[model_id] = ModelKVShape(
            model_id=model_id,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def lookup(self, model_id: str) -> ModelKVShape:
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id!r} is not registered")
        return self._registry[model_id]

    def lookup_from_datalore(
        self,
        model_id: str,
        base_url: str | None = None,
        timeout: float = 2.0,
    ) -> ModelKVShape | None:
        """Optionally query api-datalore (:9311/knowledge/lookup) for crystal KV shape metadata."""
        if model_id in self._registry:
            return self._registry[model_id]
        endpoint = (
            base_url
            or os.environ.get("DATALORE_API_URL")
            or os.environ.get("DATALORE_BASE_URL")
            or "http://127.0.0.1:9311"
        ).rstrip("/") + "/knowledge/lookup"
        payload = json.dumps({"query": model_id, "scope": "models", "limit": 1}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Datalore-Data-Mode": "ordinary-explicit",
                "X-Datalore-Ordinary-Reason": "explicit_diagnostic_export",
                "X-Datalore-Caller-Id": "sglang-universal-kv",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        for hit in data.get("results") or data.get("hits") or []:
            if not isinstance(hit, dict):
                continue
            meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else hit
            layers = meta.get("num_layers")
            kv_heads = meta.get("num_kv_heads")
            head_dim = meta.get("head_dim")
            if all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
                self.register(
                    model_id,
                    num_layers=layers,
                    num_kv_heads=kv_heads,
                    head_dim=head_dim,
                )
                return self._registry[model_id]
        return None
