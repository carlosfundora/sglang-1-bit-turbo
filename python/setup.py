from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension("sglang.sglang_rust_utils", path="sglang/rust_utils/Cargo.toml", binding=Binding.PyO3)
    ]
)
