import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'python'))

import mock_imports
import importlib.util

sys.modules["sglang"] = __import__('types').ModuleType("sglang")
sys.modules["sglang.srt"] = __import__('types').ModuleType("sglang.srt")
sys.modules["sglang.srt.entrypoints"] = __import__('types').ModuleType("sglang.srt.entrypoints")
sys.modules["sglang.srt.entrypoints.openai"] = __import__('types').ModuleType("sglang.srt.entrypoints.openai")

class MockProtocol(__import__('types').ModuleType):
    def __getattr__(self, name):
        def _mock_func(*args, **kwargs):
            class DynamicObj:
                def __init__(self, **kwargs):
                    self.continue_final_message = False
                    self.messages = []
                    self.modalities = None
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                def __getattr__(self, name):
                    return None
                def get(self, k, d=None):
                    return getattr(self, k, d)
            return DynamicObj(**kwargs)
        return _mock_func

sys.modules["sglang.srt.entrypoints.openai.protocol"] = MockProtocol("protocol")

sys.modules["sglang.srt.utils"] = __import__('types').ModuleType("sglang.srt.utils")
import sglang.srt.utils as utils
class MockImageData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
utils.ImageData = MockImageData
utils.read_system_prompt_from_file = lambda x: ""

sys.modules["sglang.test"] = __import__('types').ModuleType("sglang.test")
sys.modules["sglang.test.test_utils"] = __import__('types').ModuleType("sglang.test.test_utils")
sys.modules["sglang.test.test_utils"].CustomTestCase = __import__('unittest').TestCase
sys.modules["sglang.test.ci"] = __import__('types').ModuleType("sglang.test.ci")
sys.modules["sglang.test.ci.ci_register"] = __import__('types').ModuleType("sglang.test.ci.ci_register")
sys.modules["sglang.test.ci.ci_register"].register_cpu_ci = lambda *args, **kwargs: None

spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang.sglang_rust_utils"] = sglang_rust_utils
spec.loader.exec_module(sglang_rust_utils)

from sglang.srt.parser.conversation import Conversation, SeparatorStyle
c = Conversation("t", "", roles=("",""), sep_style=SeparatorStyle.LLAMA2, messages=[["","Hi"]])
print(c.get_prompt())
