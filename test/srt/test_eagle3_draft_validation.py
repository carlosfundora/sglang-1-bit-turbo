from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang.srt.server_args import ServerArgs


def test_eagle3_draft_validation_rejects_plain_instruct_checkpoint(monkeypatch):
    args = ServerArgs(
        model_path="target",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
    )

    monkeypatch.setattr(
        "sglang.srt.server_args.get_config",
        lambda *args, **kwargs: SimpleNamespace(architectures=["LlamaForCausalLM"]),
    )

    with pytest.raises(ValueError, match="trained EAGLE3 draft checkpoint"):
        args._validate_eagle3_draft_model()


def test_eagle3_draft_validation_accepts_eagle_config(monkeypatch):
    args = ServerArgs(
        model_path="target",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
    )

    monkeypatch.setattr(
        "sglang.srt.server_args.get_config",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            eagle_config={"use_aux_hidden_state": True},
        ),
    )

    args._validate_eagle3_draft_model()
