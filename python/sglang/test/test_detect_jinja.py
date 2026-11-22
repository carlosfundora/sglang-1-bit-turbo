import unittest
from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

class TestJinjaTemplateDetection(unittest.TestCase):
    def test_openai_format_loops(self):
        templates = [
            "{%- for content in message['content'] -%} {{ content.text }} {%- endfor -%}",
            "{% for item in msg.content %} {{ item.type }} {% endfor %}",
            "{%-for  thing in  m.content -%} ok {%-endfor-%}"
        ]
        for t in templates:
            self.assertEqual(detect_jinja_template_content_format(t), "openai")

    def test_openai_format_multimodal(self):
        templates = [
            "We have an image here",
            "There is some audio included",
            "This video is cool",
            "Computer vision is amazing"
        ]
        for t in templates:
            self.assertEqual(detect_jinja_template_content_format(t), "openai")

    def test_string_format(self):
        templates = [
            "{{ message.content }}",
            "A regular text without any variables",
            "{%- for something else in message -%} {{ something }} {%- endfor -%}",
            "{% for item in messages %} {{ item.content }} {% endfor %}"
        ]
        for t in templates:
            self.assertEqual(detect_jinja_template_content_format(t), "string")

if __name__ == "__main__":
    unittest.main()
