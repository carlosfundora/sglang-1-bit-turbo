import sys, types

sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['tqdm'] = types.ModuleType('tqdm')
sys.modules['pydantic'] = types.ModuleType('pydantic')
sys.modules['fastapi'] = types.ModuleType('fastapi')
sys.modules['uvicorn'] = types.ModuleType('uvicorn')
sys.modules['aiohttp'] = types.ModuleType('aiohttp')
sys.modules['requests'] = types.ModuleType('requests')

import unittest
import json
import importlib.util

sys.path.append("python/sglang")
import sglang_rust_utils
sys.modules['sglang.sglang_rust_utils'] = sglang_rust_utils

class TestJsonSchema(unittest.TestCase):
    def test_valid_schema(self):
        schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        })
        # Should not raise
        sglang_rust_utils.is_valid_json_schema(schema)

    def test_invalid_schema(self):
        # type array but missing items or having invalid properties
        schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "invalid_type"} # invalid type
            }
        })
        with self.assertRaises(ValueError):
            sglang_rust_utils.is_valid_json_schema(schema)

if __name__ == "__main__":
    unittest.main()
