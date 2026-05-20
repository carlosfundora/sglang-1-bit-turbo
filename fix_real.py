# Re-implement MambaRadixCache prefix matching natively using PyO3 correctly this time.
# No duplicated imports, no string matching, clean code.

# Step 1: Rust
with open("python/sglang/rust_utils/src/lib.rs", "a") as f:
    f.write("""
#[pyfunction]
pub fn mamba_match_prefix(node_key: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<usize> {
    // A clean version that doesn't just do naive list conversion to prevent extreme overhead.
    // We expect both to be tuples of integers. We can iterate directly over the PyTuple to avoid creating intermediate Vecs.
    use pyo3::types::PyTuple;

    let node_tuple = node_key.downcast::<PyTuple>()?;
    let key_tuple = key.downcast::<PyTuple>()?;

    let node_len = node_tuple.len();
    let key_len = key_tuple.len();
    let min_len = std::cmp::min(node_len, key_len);

    let mut match_len = 0;
    for i in 0..min_len {
        let n_val = node_tuple.get_item(i)?;
        let k_val = key_tuple.get_item(i)?;

        // Extract inner values
        let n: i64 = n_val.extract()?;
        let k: i64 = k_val.extract()?;

        if n == k {
            match_len += 1;
        } else {
            break;
        }
    }

    Ok(match_len)
}
""")

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    rust_content = f.read()

rust_content = rust_content.replace(
    "    m.add_function(wrap_pyfunction!(contains_chat_template, m)?)?;",
    "    m.add_function(wrap_pyfunction!(contains_chat_template, m)?)?;\n    m.add_function(wrap_pyfunction!(mamba_match_prefix, m)?)?;"
)

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(rust_content)
