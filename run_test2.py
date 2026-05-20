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

class MockTorch(types.ModuleType):
    def tensor(self, x): return x
    def arange(self, *args, **kwargs): return []
    @property
    def int64(self): return int
    def empty(self, *args, **kwargs): return []
    def cat(self, *args, **kwargs): return []

sys.modules["numpy"] = types.ModuleType("numpy")
sys.modules["numpy"].float64 = float
sys.modules["torch"] = MockTorch("torch")
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

import importlib.util
spec = importlib.util.spec_from_file_location("sglang_rust_utils", "python/sglang/sglang_rust_utils.so")
sglang_rust_utils = importlib.util.module_from_spec(spec)
sys.modules["sglang.sglang_rust_utils"] = sglang_rust_utils
spec.loader.exec_module(sglang_rust_utils)

sys.modules["sglang"] = types.ModuleType("sglang")
sys.modules["sglang.srt"] = types.ModuleType("sglang.srt")
sys.modules["sglang.srt.server_args"] = types.ModuleType("sglang.srt.server_args")
sys.modules["sglang.srt.server_args"].get_global_server_args = lambda: None

sys.modules["sglang.srt.layers"] = types.ModuleType("sglang.srt.layers")
sys.modules["sglang.srt.layers.attention"] = types.ModuleType("sglang.srt.layers.attention")
sys.modules["sglang.srt.layers.attention.fla"] = types.ModuleType("sglang.srt.layers.attention.fla")
sys.modules["sglang.srt.layers.attention.fla.chunk_delta_h"] = types.ModuleType("sglang.srt.layers.attention.fla.chunk_delta_h")
sys.modules["sglang.srt.layers.attention.fla.chunk_delta_h"].CHUNK_SIZE = 32

sys.modules["sglang.srt.mem_cache"] = types.ModuleType("sglang.srt.mem_cache")
sys.modules["sglang.srt.mem_cache.base_prefix_cache"] = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].BasePrefixCache = type('BasePrefixCache', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].DecLockRefParams = type('DecLockRefParams', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].DecLockRefResult = type('DecLockRefResult', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].EvictResult = type('EvictResult', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].EvictParams = type('EvictParams', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].InsertParams = type('InsertParams', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].InsertResult = type('InsertResult', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].IncLockRefResult = type('IncLockRefResult', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].InitLoadBackParams = type('InitLoadBackParams', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].MatchPrefixParams = type('MatchPrefixParams', (), {})
sys.modules["sglang.srt.mem_cache.base_prefix_cache"].MatchResult = type('MatchResult', (), {})

sys.modules["sglang.srt.mem_cache.allocator"] = types.ModuleType("sglang.srt.mem_cache.allocator")
sys.modules["sglang.srt.mem_cache.allocator"].TokenToKVPoolAllocator = type('TokenToKVPoolAllocator', (), {})
sys.modules["sglang.srt.mem_cache.allocator"].PagedTokenToKVPoolAllocator = type('PagedTokenToKVPoolAllocator', (), {})

sys.modules["sglang.srt.mem_cache.common"] = types.ModuleType("sglang.srt.mem_cache.common")
sys.modules["sglang.srt.mem_cache.common"].get_last_access_time = lambda: 1.0

sys.modules["sglang.srt.mem_cache.memory_pool"] = types.ModuleType("sglang.srt.mem_cache.memory_pool")
sys.modules["sglang.srt.mem_cache.memory_pool"].MambaPool = type('MambaPool', (), {})
sys.modules["sglang.srt.mem_cache.memory_pool"].ReqToTokenPool = type('ReqToTokenPool', (), {})
sys.modules["sglang.srt.mem_cache.memory_pool"].HybridReqToTokenPool = type('HybridReqToTokenPool', (), {})

sys.modules["sglang.srt.mem_cache.radix_cache"] = types.ModuleType("sglang.srt.mem_cache.radix_cache")
sys.modules["sglang.srt.mem_cache.radix_cache"].RadixKey = tuple
sys.modules["sglang.srt.mem_cache.radix_cache"].get_child_key = lambda x: x[0]
sys.modules["sglang.srt.mem_cache.radix_cache"]._key_match_page_size1 = lambda a, b: sum(1 for x, y in zip(a, b) if x == y)
sys.modules["sglang.srt.mem_cache.radix_cache"]._key_match_paged = lambda a, b: sum(1 for x, y in zip(a, b) if x == y)

sys.modules["sglang.srt.distributed"] = types.ModuleType("sglang.srt.distributed")
sys.modules["sglang.srt.distributed"].get_tensor_model_parallel_rank = lambda: 0

try:
    spec_mamba = importlib.util.spec_from_file_location("mamba_radix_cache", "python/sglang/srt/mem_cache/mamba_radix_cache.py")
    mamba_radix_cache = importlib.util.module_from_spec(spec_mamba)
    sys.modules["sglang.srt.mem_cache.mamba_radix_cache"] = mamba_radix_cache
    spec_mamba.loader.exec_module(mamba_radix_cache)
    print("MambaRadixCache loaded successfully with Rust integration!")
except Exception as e:
    print(repr(e))
    pass
