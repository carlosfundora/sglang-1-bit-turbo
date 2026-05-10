use pyo3::prelude::*;

#[pyfunction]
fn prefix_hold(text: &str, tokens: Vec<String>) -> (String, String) {
    if text.is_empty() {
        return (String::new(), String::new());
    }

    // Instead of allocating a Vec<char>, we can compute over char boundaries
    // Python string indexing is char-based, so max_hold in Python means characters.
    // If we just check `tok.starts_with(text_suffix)` we can iterate through the char indices of `text`.

    let mut char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    char_indices.push(text.len()); // for the end of the string

    let text_char_len = char_indices.len() - 1;
    let mut max_hold_chars = 0;

    for tok in tokens {
        if tok.is_empty() {
            continue;
        }

        let tok_char_len = tok.chars().count();
        let l = std::cmp::min(tok_char_len - 1, text_char_len);

        for k in (1..=l).rev() {
            let start_byte = char_indices[text_char_len - k];
            let suffix = &text[start_byte..];
            if tok.starts_with(suffix) {
                max_hold_chars = std::cmp::max(max_hold_chars, k);
                break;
            }
        }
    }

    if max_hold_chars == 0 {
        (text.to_string(), String::new())
    } else {
        let split_byte = char_indices[text_char_len - max_hold_chars];
        let (emit, hold) = text.split_at(split_byte);
        (emit.to_string(), hold.to_string())
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prefix_hold, m)?)?;
    Ok(())
}
