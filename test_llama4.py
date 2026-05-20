import sys
import types
import os
import importlib.util

sys.modules["numpy"] = types.ModuleType("numpy")
sys.path.append(os.path.join(os.getcwd(), 'python'))

spec = importlib.util.spec_from_file_location("conversation", "python/sglang/srt/parser/conversation.py")
conversation = importlib.util.module_from_spec(spec)
sys.modules["sglang.srt.parser.conversation"] = conversation
spec.loader.exec_module(conversation)

SeparatorStyle = conversation.SeparatorStyle
Conversation = conversation.Conversation

conv = Conversation(
    name="test",
    system_message="Be helpful",
    system_template="{system_message}",
    roles=("user", "assistant"),
    messages=[["user", "Hello"], ["assistant", None]],
    sep_style=SeparatorStyle.LLAMA4,
)
print("Is Rust available:", conversation.RUST_UTILS_AVAILABLE)
print(conv.get_prompt())
