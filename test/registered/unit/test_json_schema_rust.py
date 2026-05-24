import importlib.util
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, SchemaError


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


def test_rust_jsonschema_accepts_valid_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert (
        rust_utils.check_jsonschema(
            '{"type":"object","properties":{"name":{"type":"string"}}}'
        )
        is True
    )
    rust_utils.check_schema_fast(schema)


def test_rust_jsonschema_rejects_invalid_schema():
    schema = {"type": "not-a-json-schema-type"}
    with pytest.raises(ValueError):
        rust_utils.check_schema_fast(schema)
    with pytest.raises(SchemaError):
        Draft202012Validator.check_schema(schema)
