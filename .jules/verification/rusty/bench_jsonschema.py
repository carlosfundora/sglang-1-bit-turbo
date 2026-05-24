import time
import json
from jsonschema import Draft202012Validator

def bench():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }

    start = time.time()
    for _ in range(1000):
        Draft202012Validator.check_schema(schema)
    end = time.time()

    print(f"Time: {end - start:.4f} s")

bench()
