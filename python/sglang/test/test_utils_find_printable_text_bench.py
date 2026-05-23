import json
import os
import sys
import time
import types
import importlib.util

sys.path.insert(0, os.path.abspath('python'))

for m in ['numpy', 'torch', 'pydantic', 'openai', 'transformers', 'vllm', 'xformers', 'pybase64', 'requests', 'aiohttp', 'uvicorn', 'fastapi', 'urllib3', 'multipart']:
    sys.modules[m] = types.ModuleType(m)
sys.modules['pydantic.BaseModel'] = type('BaseModel', (), {})

tqdm = types.ModuleType('tqdm')
tqdm.tqdm = type('tqdm', (), {})
sys.modules['tqdm'] = tqdm

def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text_python(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    if text.endswith("\n"):
        return text
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    else:
        return text[: text.rfind(" ") + 1]

def get_implementation():
    try:
        spec_rust = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
        sglang_rust_utils = importlib.util.module_from_spec(spec_rust)
        spec_rust.loader.exec_module(sglang_rust_utils)
        if hasattr(sglang_rust_utils, "find_printable_text"):
            return sglang_rust_utils.find_printable_text, "pure rust"
    except Exception as e:
        print("Rust load error:", e)
        pass
    return find_printable_text_python, "pure python"

def bench():
    func, impl_desc = get_implementation()
    print(f"Benchmarking: {impl_desc}")

    text1 = "This is a normal English sentence "
    text2 = "This is a normal English sentence\n"
    text3 = "This is a normal English sent"
    text4 = "中文测试"
    text5 = "中文测"

    # Pre-warm
    for _ in range(100):
        func(text1)
        func(text2)
        func(text3)
        func(text4)
        func(text5)

    start_time = time.time()
    for _ in range(1000000):
        func(text1)
        func(text2)
        func(text3)
        func(text4)
        func(text5)
    duration_ms = (time.time() - start_time) * 1000

    os.makedirs(".jules/verification/rusty", exist_ok=True)

    is_after = impl_desc == "pure rust"
    filename = "after-benchmark.json" if is_after else "before-benchmark.json"

    with open(f".jules/verification/rusty/{filename}", "w") as f:
        json.dump({
            "candidate": "python/sglang/utils.py",
            "implementation": "after" if is_after else "before",
            "command": "python python/sglang/test/test_utils_find_printable_text_bench.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iterations": 1000000,
            "input_description": f"find_printable_text ({impl_desc})",
            "duration_ms": duration_ms
        }, f, indent=2)
    print("Duration:", duration_ms)

if __name__ == "__main__":
    bench()
