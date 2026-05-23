import sys, types
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['torch'] = types.ModuleType('torch')

import time
import json
from jsonschema import Draft202012Validator

schemas = [
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"}
                }
            }
        },
        "required": ["name"]
    },
    {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "number"}
            }
        }
    }
]

start = time.perf_counter()
for _ in range(5000):
    for s in schemas:
        Draft202012Validator.check_schema(s)
end = time.perf_counter()

dur = (end - start) * 1000

import json
output = {
  "candidate": "python/sglang/srt/entrypoints/openai/serving_chat.py",
  "implementation": "before",
  "command": "python3 bench_schema_run.py",
  "timestamp": "2024-05-23T08:00:00Z",
  "iterations": 5000,
  "input_description": "Valid JSON schema validation",
  "duration_ms": dur
}
with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Time taken: {dur:.4f} ms")
