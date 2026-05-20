import sys
import types
import os

sys.path.append(os.path.join(os.getcwd(), 'python'))

class MockBaseModel:
    pass

class MockPydantic(types.ModuleType):
    BaseModel = MockBaseModel

class MockTqdm(types.ModuleType):
    def tqdm(self, *args, **kwargs):
        pass

sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["tqdm"] = MockTqdm("tqdm")
sys.modules["pybase64"] = types.ModuleType("pybase64")
sys.modules["diffusers"] = types.ModuleType("diffusers")
sys.modules["transformers"] = types.ModuleType("transformers")
sys.modules["mooncake"] = types.ModuleType("mooncake")
sys.modules["requests"] = types.ModuleType("requests")
sys.modules["aiohttp"] = types.ModuleType("aiohttp")
sys.modules["uvicorn"] = types.ModuleType("uvicorn")
sys.modules["fastapi"] = types.ModuleType("fastapi")
sys.modules["pydantic"] = MockPydantic("pydantic")
sys.modules["openai"] = types.ModuleType("openai")
sys.modules["openai.types"] = types.ModuleType("openai.types")

class MockOpenAITypesResponses(types.ModuleType):
    def __getattr__(self, name):
        def _mock_func(*args, **kwargs):
            return kwargs if kwargs else args[0] if args else None
        return _mock_func
sys.modules["openai.types.responses"] = MockOpenAITypesResponses("openai.types.responses")

sys.path.insert(0, os.path.join(os.getcwd(), 'python'))

import sglang
import sglang.srt
import sglang.srt.entrypoints
import sglang.srt.entrypoints.openai

class MockProtocol(types.ModuleType):
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

sys.modules["sglang.srt.utils"] = types.ModuleType("sglang.srt.utils")
import sglang.srt.utils as utils
class MockImageData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
utils.ImageData = MockImageData
utils.read_system_prompt_from_file = lambda x: ""

sys.modules["sglang.test"] = types.ModuleType("sglang.test")
sys.modules["sglang.test.test_utils"] = types.ModuleType("sglang.test.test_utils")
sys.modules["sglang.test.test_utils"].CustomTestCase = __import__('unittest').TestCase
sys.modules["sglang.test.ci"] = types.ModuleType("sglang.test.ci")
sys.modules["sglang.test.ci.ci_register"] = types.ModuleType("sglang.test.ci.ci_register")
sys.modules["sglang.test.ci.ci_register"].register_cpu_ci = lambda *args, **kwargs: None

import importlib.util
spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang.sglang_rust_utils"] = sglang_rust_utils
spec.loader.exec_module(sglang_rust_utils)

from sglang.srt.parser.conversation import Conversation, SeparatorStyle

conv = Conversation("t", "", roles=("user","assistant"), messages=[["user","Hello"]], sep_style=SeparatorStyle.LLAMA4, sep="\n", sep2="")
print("STYLE INT:", int(conv.sep_style))
try:
    print("RUST RET:", sglang_rust_utils.conversation_get_prompt("", "", int(conv.sep_style), "\n", "", ("user", "assistant"), [["user", "Hello"]], "<image>"))
except Exception as e:
    print("RUST EXC:", repr(e))
