import time
import json

import sys, types
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['tqdm'] = types.ModuleType('tqdm')
sys.modules['pybase64'] = types.ModuleType('pybase64')
sys.modules['transformers'] = types.ModuleType('transformers')
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
sys.modules['fastapi'] = types.ModuleType('fastapi')
sys.modules['pydantic'] = types.ModuleType('pydantic')
sys.modules['diffusers'] = types.ModuleType('diffusers')

import importlib.util
spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
rust_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rust_utils)

def bench():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    schema_str = json.dumps(schema)

    start = time.time()
    for _ in range(1000):
        rust_utils.check_jsonschema(schema_str)
    end = time.time()

    print(f"Time: {end - start:.4f} s")

bench()
