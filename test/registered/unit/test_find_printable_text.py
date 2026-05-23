import unittest
import os
import sys

# Append the python directory so we can import sglang
sys.path.insert(0, os.path.abspath('python'))

class TestFindPrintableText(unittest.TestCase):
    def test_find_printable_text_parity(self):
        # We need to test the rust version against python behavior.
        # We can re-implement the python version locally to compare.

        def _is_chinese_char(cp: int):
            if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)
                or (cp >= 0x20000 and cp <= 0x2A6DF)
                or (cp >= 0x2A700 and cp <= 0x2B73F)
                or (cp >= 0x2B740 and cp <= 0x2B81F)
                or (cp >= 0x2B820 and cp <= 0x2CEAF)
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)
            ):
                return True
            return False

        def python_find_printable_text(text: str):
            if text.endswith("\n"):
                return text
            elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
                return text
            elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
                return text[:-1]
            else:
                return text[: text.rfind(" ") + 1]

        # Import the real one which should be Rust backed
        from sglang.utils import find_printable_text as rust_find_printable_text

        test_cases = [
            "This is a normal English sentence ",
            "This is a normal English sentence\n",
            "This is a normal English sent",
            "中文测试",
            "中文测",
            "English and 中文混合",
            "English and 中文混合 ",
            "A completely unrelated test",
            "",
            "\n",
            "a",
            " ",
            " a ",
            "a\n",
            "中",
            "中\n",
            "𠀀", # CJK Extension B
            "𪜀", # CJK Extension C
            "𫝀", # CJK Extension D
        ]

        for case in test_cases:
            py_res = python_find_printable_text(case)
            rs_res = rust_find_printable_text(case)
            self.assertEqual(
                py_res,
                rs_res,
                f"Mismatch for input {case!r}: Python gave {py_res!r} but Rust gave {rs_res!r}"
            )

if __name__ == '__main__':
    unittest.main()
