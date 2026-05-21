import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath('python'))

for m in ['numpy', 'torch', 'pydantic', 'openai', 'transformers', 'vllm', 'xformers', 'pybase64', 'requests', 'aiohttp', 'uvicorn', 'fastapi', 'urllib3', 'multipart', 'jinja2', 'hf_chat_utils', 'transformers.utils.chat_template_utils', 'orjson', 'partial_json_parser', 'partial_json_parser.core', 'partial_json_parser.core.options', 'psutil', 'triton', 'packaging', 'packaging.version', 'interegular', 'starlette', 'starlette.routing', 'outlines', 'outlines.fsm', 'outlines.fsm.json_schema', 'outlines_core']:
    sys.modules[m] = type('MockModule', (), {})()

sys.modules['PIL'] = type('MockModule', (), {})()
sys.modules['PIL'].Image = type('Image', (), {})
sys.modules['starlette.routing'].Mount = type('Mount', (), {})

sys.modules['torch.distributed'] = type('MockModule', (), {})()
sys.modules['tqdm'] = type('MockModule', (), {})()
sys.modules['tqdm'].tqdm = lambda x, **kwargs: x

class Allow:
    STR = 1
    OBJ = 2
    ARR = 3
    ALL = 4
sys.modules['partial_json_parser.core.options'].Allow = Allow
sys.modules['pydantic'].BaseModel = type('BaseModel', (), {})
sys.modules['pydantic'].Field = lambda *args, **kwargs: None

sys.modules['sglang.srt.entrypoints.openai.protocol'] = type('MockModule', (), {})()
sys.modules['sglang.srt.entrypoints.openai.protocol'].ChatCompletionRequest = type('ChatCompletionRequest', (), {})
sys.modules['sglang.srt.entrypoints.openai.protocol'].Tool = type('Tool', (), {})
sys.modules['sglang.srt.entrypoints.openai.protocol'].ToolChoice = type('ToolChoice', (), {})
sys.modules['sglang.srt.parser.harmony_parser'] = type('MockModule', (), {})()
sys.modules['sglang.srt.parser.harmony_parser'].HarmonyParser = type('HarmonyParser', (), {})


import importlib.util
spec_rust = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec_rust)
sys.modules['sglang.sglang_rust_utils'] = sglang_rust_utils
sys.modules['sglang_rust_utils'] = sglang_rust_utils
if sys.argv[1] == "after":
    try:
        spec_rust.loader.exec_module(sglang_rust_utils)
    except Exception:
        pass

import importlib.util
spec = importlib.util.spec_from_file_location("utils", "python/sglang/srt/function_call/utils.py")
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)
_find_common_prefix = utils._find_common_prefix

def bench(impl_name: str):
    s1 = "{" + '"a": "123", ' * 1000 + '"b": 1'
    s2 = "{" + '"a": "123", ' * 1000 + '"b": 2'

    # Warmup
    for _ in range(100):
        _find_common_prefix(s1, s2)

    start_time = time.time()
    iterations = 5000
    for _ in range(iterations):
        _find_common_prefix(s1, s2)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    filename = f".jules/verification/rusty/{impl_name}-benchmark.json"
    with open(filename, "w") as f:
        json.dump({
            "candidate": "python/sglang/srt/function_call/utils.py",
            "implementation": impl_name,
            "command": f"python python/sglang/test/test_find_common_prefix_bench.py {impl_name}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": iterations,
            "input_description": "_find_common_prefix long string",
            "duration_ms": duration_ms
        }, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench(sys.argv[1])
    else:
        print("Please provide impl name (before or after)")
