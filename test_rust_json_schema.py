import sys, types

import time
import json
sys.path.append("python/sglang")
import sglang_rust_utils
from sglang_rust_utils import is_valid_json_schema

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

schema_strs = [json.dumps(s) for s in schemas]

start = time.perf_counter()
for _ in range(5000):
    for s in schema_strs:
        is_valid_json_schema(s)
end = time.perf_counter()
dur = (end - start) * 1000

import json
output = {
  "candidate": "python/sglang/srt/entrypoints/openai/serving_chat.py",
  "implementation": "after",
  "command": "python3 bench_schema_run.py",
  "timestamp": "2024-05-23T08:00:00Z",
  "iterations": 5000,
  "input_description": "Valid JSON schema validation",
  "duration_ms": dur
}
with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Time taken: {dur:.4f} ms")
