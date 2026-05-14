use pyo3::prelude::*;

#[pyfunction]
fn trim_overlap(existing_text: &str, new_chunk: &str) -> String {
    let max_possible = existing_text.len().min(new_chunk.len());
    let mut max_overlap = 0;

    for i in (1..=max_possible).rev() {
        if new_chunk.is_char_boundary(i) && existing_text.ends_with(&new_chunk[..i]) {
            max_overlap = i;
            break;
        }
    }

    new_chunk[max_overlap..].to_string()
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    Ok(())
}
