import sys
import types
import os
import importlib.util

sys.modules["numpy"] = types.ModuleType("numpy")
sys.path.append(os.path.join(os.getcwd(), 'python'))

spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang_rust_utils"] = sglang_rust_utils
spec.loader.exec_module(sglang_rust_utils)

try:
    print(sglang_rust_utils.conversation_get_prompt("", "", 9, "\n", None, ("user", "assistant"), [["user", "Hello"]], "<image>"))
except Exception as e:
    print(repr(e))
