import re

with open("python/sglang/srt/parser/conversation.py", "r") as f:
    content = f.read()

enum_match = re.search(r"class SeparatorStyle\(IntEnum\):(.*?)class", content, re.DOTALL)
if enum_match:
    lines = enum_match.group(1).splitlines()
    val = 1
    for line in lines:
        if "=" in line and "auto()" in line:
            name = line.split("=")[0].strip()
            print(f"{name} = {val}")
            val += 1
