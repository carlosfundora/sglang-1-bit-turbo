use ignore::WalkBuilder;
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

#[pyfunction]
fn find_files(path: &str) -> PyResult<Vec<String>> {
    let mut files = Vec::new();
    let walker = WalkBuilder::new(path)
        .follow_links(true) // Crucial for HF cache
        .hidden(false)      // Include hidden files if needed, but usually we filter in Python
        .git_ignore(false)  // Don't skip files based on .gitignore for model loading
        .build();

    for result in walker {
        match result {
            Ok(entry) => {
                if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    if let Some(s) = entry.path().to_str() {
                        files.append(&mut vec![s.to_string()]);
                    }
                }
            }
            Err(err) => return Err(pyo3::exceptions::PyIOError::new_err(err.to_string())),
        }
    }

    Ok(files)
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
    Ok(())
}
