import sys
import types
import os
import time
import json
import importlib.util

sys.modules["numpy"] = types.ModuleType("numpy")

sys.path.append(os.path.join(os.getcwd(), 'python'))

sys.modules["sglang"] = types.ModuleType("sglang")
sys.modules["sglang.srt"] = types.ModuleType("sglang.srt")
sys.modules["sglang.srt.entrypoints"] = types.ModuleType("sglang.srt.entrypoints")
sys.modules["sglang.srt.entrypoints.openai"] = types.ModuleType("sglang.srt.entrypoints.openai")
sys.modules["sglang.srt.entrypoints.openai.protocol"] = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
import sglang.srt.entrypoints.openai.protocol as protocol
protocol.ChatCompletionRequest = type('ChatCompletionRequest', (), {'continue_final_message': False})

sys.modules["sglang.srt.utils"] = types.ModuleType("sglang.srt.utils")
import sglang.srt.utils as utils
utils.ImageData = type('ImageData', (), {})
utils.read_system_prompt_from_file = lambda x: ""

spec = importlib.util.spec_from_file_location("conversation", "python/sglang/srt/parser/conversation.py")
conversation = importlib.util.module_from_spec(spec)
sys.modules["sglang.srt.parser.conversation"] = conversation
spec.loader.exec_module(conversation)

Conversation = conversation.Conversation
SeparatorStyle = conversation.SeparatorStyle

def run_bench():
    conv = Conversation(
        name="test",
        system_template="{system_message}",
        system_message="You are a helpful assistant.",
        roles=("USER", "ASSISTANT"),
        messages=[],
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
    for i in range(100):
        conv.append_message(conv.roles[i % 2], f"Message {i}")

    start_time = time.time()
    for _ in range(5000):
        _ = conv.get_prompt()
    end_time = time.time()

    duration_ms = (end_time - start_time) * 1000

    result = {
        "candidate": "python/sglang/srt/parser/conversation.py",
        "implementation": "before",
        "command": "python3 bench_conv.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": 5000,
        "input_description": "100 messages prompt generation",
        "duration_ms": duration_ms
    }

    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    run_bench()
