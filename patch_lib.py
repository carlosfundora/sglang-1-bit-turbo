import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

new_func = """
#[pyfunction]
fn is_valid_json_schema(schema_str: &str) -> PyResult<()> {
    let schema_json: serde_json::Value = serde_json::from_str(schema_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;
    match jsonschema::JSONSchema::options().compile(&schema_json) {
        Ok(_) => Ok(()),
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON Schema: {}", e))),
    }
}
"""

content = content.replace("#[pymodule]", new_func + "\n#[pymodule]")
content = content.replace("m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;", "m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;\n    m.add_function(wrap_pyfunction!(is_valid_json_schema, m)?)?;")

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(content)
