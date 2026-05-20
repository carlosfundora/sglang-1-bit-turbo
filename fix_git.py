import subprocess
import time
subprocess.check_call(["git", "restore", "python/sglang/rust_utils/src/lib.rs"])
time.sleep(0.5)

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

rust_code = """
#[pyfunction]
pub fn mamba_split_node(child_key_prefix: Vec<i64>, split_len: usize) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let new_child_key = child_key_prefix[split_len..].to_vec();
    let new_node_key = child_key_prefix[..split_len].to_vec();
    Ok((new_node_key, new_child_key))
}

#[pyfunction]
pub fn mamba_match_prefix(node_key: Vec<i64>, key: Vec<i64>) -> usize {
    let mut match_len = 0;
    for (a, b) in node_key.iter().zip(key.iter()) {
        if a == b {
            match_len += 1;
        } else {
            break;
        }
    }
    match_len
}
"""

content = content.replace("m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;",
"m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;\n    m.add_function(wrap_pyfunction!(mamba_split_node, m)?)?;\n    m.add_function(wrap_pyfunction!(mamba_match_prefix, m)?)?;")

content = content + "\n\n" + rust_code

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
