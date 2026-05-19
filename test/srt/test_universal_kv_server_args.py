import pathlib


def test_server_args_exposes_universal_attention_backend_choice():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "python" / "sglang" / "srt" / "server_args.py").read_text()
    assert '"universal_broker"' in source


def test_server_args_exposes_hybrid_kv_dtypes():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "python" / "sglang" / "srt" / "server_args.py").read_text()
    assert '"rq3_hybrid"' in source
    assert '"univ_rq3"' in source


def test_attention_registry_has_universal_broker():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "attention_registry.py"
    ).read_text()
    assert '@register_attention_backend("universal_broker")' in source


def test_universal_backend_wrapper_exists():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "universal_broker_backend.py"
    ).read_text()
    assert "class UniversalBrokerAttnBackend" in source
    assert "universal_kv_gpu_capacity_mb" in source
    assert "universal_kv_ram_capacity_mb" in source


def test_server_args_exposes_universal_broker_capacity_flags():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "python" / "sglang" / "srt" / "server_args.py").read_text()
    assert '"--universal-kv-gpu-capacity-mb"' in source
    assert '"--universal-kv-ram-capacity-mb"' in source
    assert '"--universal-kv-block-size"' in source
    assert '"--universal-kv-hot-importance-threshold"' in source


def test_server_args_autoselects_universal_backend_for_hybrid_mode():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / "python" / "sglang" / "srt" / "server_args.py").read_text()
    assert "Auto-select attention backend 'universal_broker'" in source
