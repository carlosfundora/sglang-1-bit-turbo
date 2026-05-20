with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

content = content.replace("m.add_class::<RustReasoningState>()?;\n    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;", "m.add_class::<RustReasoningState>()?;\n    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;\n    m.add_function(wrap_pyfunction!(mamba_match_prefix, m)?)?;")

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
