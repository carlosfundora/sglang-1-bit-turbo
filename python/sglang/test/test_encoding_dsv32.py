import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath('python'))

# Mock missing dependencies
import types
for m in ['numpy', 'torch', 'pydantic', 'openai', 'transformers', 'vllm', 'xformers', 'pybase64', 'requests', 'aiohttp', 'uvicorn', 'fastapi', 'urllib3', 'multipart', 'jinja2', 'hf_chat_utils', 'transformers.utils.chat_template_utils', 'orjson', 'partial_json_parser', 'partial_json_parser.core', 'partial_json_parser.core.options', 'psutil', 'triton', 'packaging', 'packaging.version', 'interegular', 'starlette', 'starlette.routing', 'outlines', 'outlines.fsm', 'outlines.fsm.json_schema', 'outlines_core']:
    sys.modules[m] = type('MockModule', (), {})()

sys.modules['PIL'] = type('MockModule', (), {})()
sys.modules['PIL'].Image = type('Image', (), {})
sys.modules['starlette.routing'].Mount = type('Mount', (), {})
sys.modules['torch.distributed'] = type('MockModule', (), {})()
sys.modules['tqdm'] = type('MockModule', (), {})()
sys.modules['tqdm'].tqdm = lambda x, **kwargs: x
sys.modules['pydantic'].BaseModel = type('BaseModel', (), {})
sys.modules['pydantic'].Field = lambda *args, **kwargs: None

sys.modules['sglang.srt.entrypoints.openai.protocol'] = type('MockModule', (), {})()
sys.modules['sglang.srt.entrypoints.openai.protocol'].ChatCompletionRequest = type('ChatCompletionRequest', (), {})
sys.modules['sglang.srt.entrypoints.openai.protocol'].Tool = type('Tool', (), {})
sys.modules['sglang.srt.entrypoints.openai.protocol'].ToolChoice = type('ToolChoice', (), {})
sys.modules['sglang.srt.parser.harmony_parser'] = type('MockModule', (), {})()
sys.modules['sglang.srt.parser.harmony_parser'].HarmonyParser = type('HarmonyParser', (), {})


# Try injecting Rust extension if testing "after"
if len(sys.argv) > 1 and sys.argv[1] == "after":
    import importlib.util
    try:
        spec_rust = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
        sglang_rust_utils = importlib.util.module_from_spec(spec_rust)
        sys.modules['sglang.sglang_rust_utils'] = sglang_rust_utils
        sys.modules['sglang_rust_utils'] = sglang_rust_utils
        spec_rust.loader.exec_module(sglang_rust_utils)
    except Exception as e:
        print("Could not load rust extension:", e)

import importlib.util
spec = importlib.util.spec_from_file_location("dsv32", "python/sglang/srt/entrypoints/openai/encoding_dsv32.py")
dsv32 = importlib.util.module_from_spec(spec)
sys.modules["dsv32"] = dsv32
spec.loader.exec_module(dsv32)

def test_correctness():
    dsml_token = "｜DSML｜"
    text = "Here is some thinking or summary content."
    text += f"\n\n<{dsml_token}function_calls>\n"

    text += f"<{dsml_token}invoke name=\"tool_0\">\n"
    text += f'<{dsml_token}parameter name="param_0" string="true">value_0</{dsml_token}parameter>\n'
    text += f'</{dsml_token}invoke>\n'

    text += f"</{dsml_token}function_calls>"

    res = dsv32.parse_message_from_completion_text(text, "none")
    assert res['role'] == 'assistant'
    assert res['content'] == 'Here is some thinking or summary content.'
    assert res['tool_calls'][0]['function']['name'] == 'tool_0'
    assert json.loads(res['tool_calls'][0]['function']['arguments']) == {'param_0': 'value_0'}

def benchmark():
    # Let's mock a long tool calling text
    dsml_token = dsv32.dsml_token
    text = "Here is some thinking or summary content."
    text += f"\n\n<{dsml_token}function_calls>\n"

    # Add multiple tool calls
    for i in range(50):
        text += f"<{dsml_token}invoke name=\"tool_{i}\">\n"
        for j in range(10):
            text += f'<{dsml_token}parameter name="param_{j}" string="true">value_{j}</{dsml_token}parameter>\n'
        text += f'</{dsml_token}invoke>\n'

    text += f"</{dsml_token}function_calls>"

    # Pre-warm
    for _ in range(10):
        dsv32.parse_message_from_completion_text(text, "none")

    start = time.time()
    iterations = 1000
    for _ in range(iterations):
        dsv32.parse_message_from_completion_text(text, "none")
    duration = (time.time() - start) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    impl_name = sys.argv[1] if len(sys.argv) > 1 else "before"
    with open(f".jules/verification/rusty/{impl_name}-benchmark.json", "w") as f:
        json.dump({
            "candidate": "python/sglang/srt/entrypoints/openai/encoding_dsv32.py",
            "implementation": impl_name,
            "command": f"python python/sglang/test/test_encoding_dsv32.py {impl_name}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": iterations,
            "input_description": "parse_message_from_completion_text with 50 tool calls",
            "duration_ms": duration
        }, f, indent=2)

if __name__ == "__main__":
    test_correctness()
    benchmark()
    print("Done")

def test_correctness_nested_json():
    dsml_token = "｜DSML｜"
    text = "Here is some thinking or summary content."
    text += f"\n\n<{dsml_token}function_calls>\n"

    text += f"<{dsml_token}invoke name=\"tool_1\">\n"
    text += f'<{dsml_token}parameter name="param_0" string="false">[1, 2, 3]</{dsml_token}parameter>\n'
    text += f'<{dsml_token}parameter name="param_1" string="false">{{"key": "value"}}</{dsml_token}parameter>\n'
    text += f'</{dsml_token}invoke>\n'

    text += f"</{dsml_token}function_calls>"

    res = dsv32.parse_message_from_completion_text(text, "none")
    assert res['role'] == 'assistant'
    assert res['content'] == 'Here is some thinking or summary content.'
    assert res['tool_calls'][0]['function']['name'] == 'tool_1'
    args = json.loads(res['tool_calls'][0]['function']['arguments'])
    assert args == {'param_0': [1, 2, 3], 'param_1': {'key': 'value'}}

test_correctness_nested_json()
print("Nested JSON test passed")
