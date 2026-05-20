with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# Make sure it's exported in the pymodule!
content = content.replace("m.add_function(wrap_pyfunction!(mamba_match_prefix, m)?)?;", "")
content = content.replace("m.add_function(wrap_pyfunction!(contains_chat_template, m)?)?;", "m.add_function(wrap_pyfunction!(contains_chat_template, m)?)?;\n    m.add_function(wrap_pyfunction!(mamba_match_prefix, m)?)?;")

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
