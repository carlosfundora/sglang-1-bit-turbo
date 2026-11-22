import json
import os
import sys
import time
import types
import importlib.util

sys.path.insert(0, os.path.abspath('python'))

for m in ['numpy', 'torch', 'pydantic', 'openai', 'transformers', 'vllm', 'xformers', 'pybase64', 'requests', 'aiohttp', 'uvicorn', 'fastapi', 'urllib3', 'multipart', 'jinja2', 'hf_chat_utils', 'sglang.srt.utils', 'transformers.utils.chat_template_utils']:
    sys.modules[m] = types.ModuleType(m)
sys.modules['pydantic.BaseModel'] = type('BaseModel', (), {})

spec_rust = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec_rust)
sys.modules['sglang.sglang_rust_utils'] = sglang_rust_utils
spec_rust.loader.exec_module(sglang_rust_utils)

sys.modules['sglang.srt.entrypoints.openai.protocol'] = types.ModuleType('sglang.srt.entrypoints.openai.protocol')
sys.modules['sglang.srt.entrypoints.openai.protocol'].ChatCompletionRequest = type('ChatCompletionRequest', (), {})
sys.modules['sglang.srt.parser.harmony_parser'] = types.ModuleType('sglang.srt.parser.harmony_parser')
sys.modules['sglang.srt.parser.harmony_parser'].HarmonyParser = type('HarmonyParser', (), {})

spec = importlib.util.spec_from_file_location("reasoning_parser", "python/sglang/srt/parser/reasoning_parser.py")
reasoning_parser = importlib.util.module_from_spec(spec)
sys.modules["reasoning_parser"] = reasoning_parser
spec.loader.exec_module(reasoning_parser)

def test_correctness():
    parser = reasoning_parser.BaseReasoningFormatDetector("<think>", "</think>")

    # 1. No think token
    res = parser.detect_and_parse("Just normal text.")
    assert res.normal_text == "Just normal text.", f"Got {res.normal_text!r}"
    assert res.reasoning_text == "", f"Got {res.reasoning_text!r}"

    # 2. Only think token start
    res = parser.detect_and_parse("<think> Thinking about things...")
    assert res.normal_text == "", f"Got {res.normal_text!r}"
    assert res.reasoning_text == "Thinking about things...", f"Got {res.reasoning_text!r}"

    # 3. Think token start and end
    res = parser.detect_and_parse("<think> I am thinking. </think> Now I am speaking.")
    assert res.normal_text == "Now I am speaking.", f"Got {res.normal_text!r}"
    assert res.reasoning_text == "I am thinking. ", f"Got {res.reasoning_text!r}"

    print("Correctness tests passed!")

def bench_before():
    text = "<think> This is some reasoning text that goes on for a bit. And then </think> And some normal text."

    # Pre-warm
    parser = reasoning_parser.BaseReasoningFormatDetector("<think>", "</think>")
    parser.rust_state = None
    for _ in range(100):
        parser.detect_and_parse(text)

    start_time = time.time()
    for _ in range(100000):
        parser.detect_and_parse(text)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump({
            "candidate": "python/sglang/srt/parser/reasoning_parser.py",
            "implementation": "before",
            "command": "python python/sglang/test/test_reasoning_parser_bench.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": 100000,
            "input_description": "detect_and_parse (pure python)",
            "duration_ms": duration_ms
        }, f, indent=2)

def bench_after():
    text = "<think> This is some reasoning text that goes on for a bit. And then </think> And some normal text."

    # Pre-warm
    parser = reasoning_parser.BaseReasoningFormatDetector("<think>", "</think>")
    assert parser.rust_state is not None, "Rust state is None!"
    for _ in range(100):
        parser.detect_and_parse(text)

    start_time = time.time()
    for _ in range(100000):
        parser.detect_and_parse(text)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
        json.dump({
            "candidate": "python/sglang/srt/parser/reasoning_parser.py",
            "implementation": "after",
            "command": "python python/sglang/test/test_reasoning_parser_bench.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": 100000,
            "input_description": "detect_and_parse (pure rust)",
            "duration_ms": duration_ms
        }, f, indent=2)

if __name__ == "__main__":
    test_correctness()
    bench_before()
    bench_after()
    print("Done")
