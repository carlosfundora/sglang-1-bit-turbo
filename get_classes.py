import re

with open("python/sglang/srt/parser/reasoning_parser.py", "r") as f:
    content = f.read()

print("Detect and parse:", bool(re.search("def detect_and_parse", content)))
print("Parse streaming increment:", bool(re.search("def parse_streaming_increment", content)))
