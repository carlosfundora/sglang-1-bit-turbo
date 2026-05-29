import json
import os
import sys
import time
import types
import importlib.util

sys.path.insert(0, os.path.abspath('python'))

for m in ['numpy', 'torch', 'pydantic', 'openai', 'transformers', 'vllm', 'xformers', 'pybase64', 'requests', 'aiohttp', 'uvicorn', 'fastapi', 'urllib3', 'multipart', 'jinja2', 'hf_chat_utils', 'sglang.srt.utils', 'transformers.utils.chat_template_utils', 'httpx', 'tqdm', 'datasets']:
    sys.modules[m] = types.ModuleType(m)

sys.modules['pydantic'].BaseModel = type('BaseModel', (), {})
sys.modules['httpx'].Client = type('Client', (), {})
sys.modules['openai'].OpenAI = type('OpenAI', (), {})
sys.modules['transformers'].AutoTokenizer = type('AutoTokenizer', (), {})
sys.modules['tqdm'].tqdm = type('tqdm', (), {})

jinja2 = sys.modules['jinja2']
class DummyEnv:
    def __init__(self, *args, **kwargs):
        self.globals = {}
jinja2.Environment = DummyEnv
jinja2.BaseLoader = type('BaseLoader', (), {})
jinja2.StrictUndefined = type('StrictUndefined', (), {})
jinja2.select_autoescape = lambda x: x

spec_rust = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec_rust)
sys.modules['sglang.sglang_rust_utils'] = sglang_rust_utils
spec_rust.loader.exec_module(sglang_rust_utils)

spec = importlib.util.spec_from_file_location("simple_eval_common", "python/sglang/test/simple_eval_common.py")
simple_eval_common = importlib.util.module_from_spec(spec)
sys.modules["sglang.test.simple_eval_common"] = simple_eval_common
spec.loader.exec_module(simple_eval_common)

spec2 = importlib.util.spec_from_file_location("simple_eval_longbench_v2", "python/sglang/test/simple_eval_longbench_v2.py")
simple_eval_longbench_v2 = importlib.util.module_from_spec(spec2)
sys.modules["simple_eval_longbench_v2"] = simple_eval_longbench_v2
spec2.loader.exec_module(simple_eval_longbench_v2)

def test_correctness():
    # 1. No answer
    res = simple_eval_longbench_v2.extract_longbench_v2_answer("Just normal text.")
    assert res is None, f"Got {res!r}"

    # 2. First Regex
    res = simple_eval_longbench_v2.extract_longbench_v2_answer("The correct answer is (B)")
    assert res == "B", f"Got {res!r}"

    # 3. Second Regex
    res = simple_eval_longbench_v2.extract_longbench_v2_answer("The correct answer is C")
    assert res == "C", f"Got {res!r}"

    # 4. Third Regex
    res = simple_eval_longbench_v2.extract_longbench_v2_answer("Answer: D")
    assert res == "D", f"Got {res!r}"

    # 5. Fourth Regex
    res = simple_eval_longbench_v2.extract_longbench_v2_answer("answer is A")
    assert res == "A", f"Got {res!r}"

    print("Correctness tests passed!")

def bench_before():
    text = "Some reasoning about the problem. I think The correct answer is (C)"

    # disable rust
    simple_eval_longbench_v2.rust_extract = None

    # Pre-warm
    for _ in range(100):
        simple_eval_longbench_v2.extract_longbench_v2_answer(text)

    start_time = time.time()
    for _ in range(100000):
        simple_eval_longbench_v2.extract_longbench_v2_answer(text)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump({
            "candidate": "python/sglang/test/simple_eval_longbench_v2.py",
            "implementation": "before",
            "command": "python python/sglang/test/test_longbench_v2_bench.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": 100000,
            "input_description": "extract_longbench_v2_answer (pure python)",
            "duration_ms": duration_ms
        }, f, indent=2)

def bench_after():
    text = "Some reasoning about the problem. I think The correct answer is (C)"

    # enable rust
    simple_eval_longbench_v2.rust_extract = sglang_rust_utils.py_extract_longbench_v2_answer

    # Pre-warm
    for _ in range(100):
        simple_eval_longbench_v2.extract_longbench_v2_answer(text)

    start_time = time.time()
    for _ in range(100000):
        simple_eval_longbench_v2.extract_longbench_v2_answer(text)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)
    with open(".jules/verification/rusty/after-benchmark.json", "w") as f:
        json.dump({
            "candidate": "python/sglang/test/simple_eval_longbench_v2.py",
            "implementation": "after",
            "command": "python python/sglang/test/test_longbench_v2_bench.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": 100000,
            "input_description": "extract_longbench_v2_answer (pure rust)",
            "duration_ms": duration_ms
        }, f, indent=2)

if __name__ == "__main__":
    test_correctness()
    bench_before()
    bench_after()
    print("Done")
