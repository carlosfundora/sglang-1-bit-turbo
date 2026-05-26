import sys, types

class MockBaseModel:
    pass

class MockPydantic(types.ModuleType):
    BaseModel = MockBaseModel

sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["tqdm"] = types.ModuleType("tqdm")
sys.modules["pybase64"] = types.ModuleType("pybase64")
sys.modules["diffusers"] = types.ModuleType("diffusers")
sys.modules["transformers"] = types.ModuleType("transformers")
sys.modules["mooncake"] = types.ModuleType("mooncake")
sys.modules["requests"] = types.ModuleType("requests")
sys.modules["aiohttp"] = types.ModuleType("aiohttp")
sys.modules["uvicorn"] = types.ModuleType("uvicorn")
sys.modules["fastapi"] = types.ModuleType("fastapi")
sys.modules["pydantic"] = MockPydantic("pydantic")

import sglang.srt.parser.reasoning_parser
print("Imports working")
