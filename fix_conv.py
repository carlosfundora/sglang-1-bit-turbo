import sys

with open("python/sglang/srt/parser/conversation.py", "r") as f:
    content = f.read()

import_str = """
try:
    from sglang.sglang_rust_utils import conversation_get_prompt
    RUST_UTILS_AVAILABLE = True
except ImportError:
    RUST_UTILS_AVAILABLE = False
"""

# Find where to put it
lines = content.splitlines()
idx = 0
for i, line in enumerate(lines):
    if line.startswith("from sglang.srt.utils"):
        idx = i + 1
        break

lines.insert(idx, import_str)
content = "\n".join(lines)

replacement = """    def get_prompt(self) -> str:
        \"\"\"Get the prompt for generation.\"\"\"
        if RUST_UTILS_AVAILABLE:
            try:
                sep_style_int = int(self.sep_style)
                return conversation_get_prompt(
                    self.system_template,
                    self.system_message,
                    sep_style_int,
                    self.sep,
                    self.sep2,
                    self.roles,
                    [[r, m if m is not None else ""] for r, m in self.messages],
                    self.image_token
                )
            except ValueError:
                pass

        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:"""

content = content.replace("""    def get_prompt(self) -> str:
        \"\"\"Get the prompt for generation.\"\"\"


        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:""", replacement)

with open("python/sglang/srt/parser/conversation.py", "w") as f:
    f.write(content)
