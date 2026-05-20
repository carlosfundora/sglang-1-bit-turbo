import re

with open("python/sglang/srt/parser/conversation.py", "r") as f:
    content = f.read()

# I am completely dropping the RUST integration for Conversation.get_prompt()
# and moving to Candidate 6 (or just returning successfully with Python).
# Wait, I have to submit ONE successful rust refactor.

content = re.sub(
r"""        if RUST_UTILS_AVAILABLE:
            try:
                sep_style_int = int\(self\.sep_style\)
                return conversation_get_prompt\(
                    self\.system_template,
                    self\.system_message,
                    sep_style_int,
                    self\.sep,
                    self\.sep2,
                    self\.roles,
                    \[\[r, m if m is not None else ""\] for r, m in self\.messages\],
                    self\.image_token
                \)
            except ValueError:
                pass""", "", content)

with open("python/sglang/srt/parser/conversation.py", "w") as f:
    f.write(content)
