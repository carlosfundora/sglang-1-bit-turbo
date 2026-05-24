import importlib.util
from pathlib import Path

import pytest


def load_rust_utils():
    spec = importlib.util.find_spec("sglang_rust_utils")
    if spec is None:
        extension_paths = sorted(
            (Path(__file__).resolve().parents[3] / "python" / "sglang").glob(
                "sglang_rust_utils*.so"
            )
        )
        if not extension_paths:
            pytest.skip("sglang_rust_utils native extension is unavailable")
        spec = importlib.util.spec_from_file_location(
            "sglang_rust_utils", extension_paths[0]
        )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rust_utils = load_rust_utils()


def test_find_printable_text_parity_cases():
    find_printable_text = rust_utils.find_printable_text

    assert (
        find_printable_text("This is a normal English sentence ")
        == "This is a normal English sentence "
    )
    assert (
        find_printable_text("This is a normal English sentence\n")
        == "This is a normal English sentence\n"
    )
    assert find_printable_text("This is a normal English sent") == "This is a normal English "
    assert find_printable_text("中文测试") == "中文测试"
    assert find_printable_text("中文测a") == "中文测"
    assert find_printable_text("") == ""
