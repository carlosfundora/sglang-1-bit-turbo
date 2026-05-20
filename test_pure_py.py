import sys
import os
import types

sys.path.append(os.path.join(os.getcwd(), 'python'))
sys.modules["numpy"] = types.ModuleType("numpy")

import importlib.util
spec = importlib.util.spec_from_file_location("conversation", "python/sglang/srt/parser/conversation.py")
conversation = importlib.util.module_from_spec(spec)
sys.modules["sglang.srt.parser.conversation"] = conversation

try:
    spec.loader.exec_module(conversation)
except Exception as e:
    pass

with open("python/sglang/srt/parser/conversation.py", "r") as f:
    for line in f:
        if "LLAMA4" in line:
            print(line.strip())
