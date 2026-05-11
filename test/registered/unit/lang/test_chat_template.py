import unittest
import importlib.util
import sys
import os

# Use the absolute path for the module
module_path = os.path.abspath("python/sglang/lang/chat_template.py")
spec = importlib.util.spec_from_file_location("chat_template", module_path)
chat_template = importlib.util.module_from_spec(spec)
sys.modules["chat_template"] = chat_template
spec.loader.exec_module(chat_template)

ChatTemplate = chat_template.ChatTemplate
ChatTemplateStyle = chat_template.ChatTemplateStyle
chat_template_registry = chat_template.chat_template_registry
matching_function_registry = chat_template.matching_function_registry
register_chat_template = chat_template.register_chat_template
register_chat_template_matching_function = chat_template.register_chat_template_matching_function
get_chat_template = chat_template.get_chat_template
get_chat_template_by_model_path = chat_template.get_chat_template_by_model_path

class TestChatTemplate(unittest.TestCase):
    def setUp(self):
        # Save original registries
        self.orig_chat_template_registry = chat_template_registry.copy()
        self.orig_matching_function_registry = list(matching_function_registry)

    def tearDown(self):
        # Restore original registries
        chat_template_registry.clear()
        chat_template_registry.update(self.orig_chat_template_registry)
        matching_function_registry.clear()
        matching_function_registry.extend(self.orig_matching_function_registry)

    def test_get_prompt_plain(self):
        template = ChatTemplate(
            name="test_plain",
            default_system_prompt="System default",
            role_prefix_and_suffix={
                "system": ("S:", "\n"),
                "user": ("U:", "\n"),
                "assistant": ("A:", "\n"),
            },
            style=ChatTemplateStyle.PLAIN,
        )
        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        prompt = template.get_prompt(messages)
        expected = "S:System default\nU:Hello\nA:Hi\n"
        self.assertEqual(prompt, expected)

    def test_get_prompt_llama2(self):
        template = ChatTemplate(
            name="test_llama2",
            default_system_prompt=None,
            role_prefix_and_suffix={
                "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
                "user": ("[INST] ", " [/INST]"),
                "assistant": ("", " </s><s>"),
            },
            style=ChatTemplateStyle.LLAMA2,
        )
        # Test system message at the beginning
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = template.get_prompt(messages)
        # LLAMA2 style: user prefix + system prefix + content + system suffix + user suffix
        self.assertIn("[INST] <<SYS>>\nSystem prompt\n<</SYS>>\n\nHello [/INST]", prompt)

        # Test user message alone
        messages = [{"role": "user", "content": "Hello"}]
        prompt = template.get_prompt(messages)
        self.assertEqual(prompt, "[INST] Hello [/INST]")

        # Test multi-turn
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Next"},
        ]
        prompt = template.get_prompt(messages)
        # In Llama 2, subsequent user turns also have [INST]
        expected = "[INST] Hello [/INST]Hi </s><s>[INST] Next [/INST]"
        self.assertEqual(prompt, expected)

    def test_registration(self):
        name = "custom_template"
        template = ChatTemplate(
            name=name,
            default_system_prompt=None,
            role_prefix_and_suffix={"user": ("U:", ""), "assistant": ("A:", "")},
        )
        register_chat_template(template)
        self.assertIn(name, chat_template_registry)
        self.assertEqual(get_chat_template(name), template)

    def test_matching_function(self):
        def match_custom(model_path):
            if "custom-model" in model_path:
                return "test_plain"
            return None

        register_chat_template_matching_function(match_custom)

        # We need a "test_plain" in registry for this to work
        template = ChatTemplate(
            name="test_plain",
            default_system_prompt=None,
            role_prefix_and_suffix={"user": ("U:", ""), "assistant": ("A:", "")},
        )
        register_chat_template(template)

        self.assertEqual(get_chat_template_by_model_path("path/to/custom-model"), template)

        # Test default fallback
        self.assertEqual(get_chat_template_by_model_path("other"), get_chat_template("default"))

if __name__ == "__main__":
    unittest.main()
