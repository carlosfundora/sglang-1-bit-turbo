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

#[pyclass]
pub struct RustReasoningState {
    #[pyo3(get)]
    pub buffer: String,
    #[pyo3(get)]
    pub in_reasoning: bool,
    #[pyo3(get)]
    pub stripped_think_start: bool,
}

#[pymethods]
impl RustReasoningState {
    #[new]
    fn new(in_reasoning: bool, stripped_think_start: bool, buffer: String) -> Self {
        RustReasoningState {
            in_reasoning,
            stripped_think_start,
            buffer,
        }
    }

    fn parse_streaming_increment(
        &mut self,
        new_text: &str,
        think_start_token: &str,
        think_end_token: &str,
        tool_start_token: Option<&str>,
        stream_reasoning: bool,
    ) -> PyResult<(Option<String>, Option<String>)> {
        self.buffer.push_str(new_text);

        let mut tokens_to_check = vec![think_start_token, think_end_token];
        if let Some(token) = tool_start_token {
            tokens_to_check.push(token);
        }

        let mut is_prefix = false;
        let buf_len = self.buffer.len();

        // Check if any suffix of the buffer is a prefix of any token
        for token in &tokens_to_check {
            let token_len = token.len();
            for k in (1..=std::cmp::min(token_len - 1, buf_len)).rev() {
                if self.buffer.is_char_boundary(buf_len - k) {
                    let suffix = &self.buffer[buf_len - k..];
                    if token.starts_with(suffix) {
                        is_prefix = true;
                        break;
                    }
                }
            }
            if is_prefix {
                break;
            }
        }

        if is_prefix {
            return Ok((None, None));
        }

        // Strip `<think>` token if present
        if !self.stripped_think_start && self.buffer.contains(think_start_token) {
            if let Some(start_idx) = self.buffer.find(think_start_token) {
                let normal_text = self.buffer[..start_idx].to_string();
                self.buffer = self.buffer[start_idx + think_start_token.len()..].to_string();
                self.stripped_think_start = true;
                self.in_reasoning = true;

                // If there's more logic for the end token, we return the normal text early and process reasoning next cycle
                // or we process the rest of the buffer right away. We'll do it right away.
                let mut ret_normal = Some(normal_text);
                let mut ret_reasoning = None;

                if self.buffer.contains(think_end_token) {
                    if let Some(end_idx) = self.buffer.find(think_end_token) {
                        let reasoning_text = self.buffer[..end_idx].to_string();
                        self.in_reasoning = false;
                        let after_normal = self.buffer[end_idx + think_end_token.len()..].to_string();
                        self.buffer = "".to_string();

                        let combined_normal = format!("{}{}", ret_normal.unwrap_or_default(), after_normal);
                        ret_normal = if combined_normal.is_empty() { None } else { Some(combined_normal) };
                        ret_reasoning = Some(reasoning_text.trim_end().to_string());
                        return Ok((ret_normal, ret_reasoning));
                    }
                }

                if stream_reasoning && !self.buffer.is_empty() {
                    ret_reasoning = Some(self.buffer.clone());
                    self.buffer.clear();
                }

                let final_normal = ret_normal.filter(|s| !s.is_empty());
                let final_reasoning = ret_reasoning.filter(|s| !s.is_empty());
                return Ok((final_normal, final_reasoning));
            }
        }

        // Handle end of reasoning block
        if self.in_reasoning && self.buffer.contains(think_end_token) {
            if let Some(end_idx) = self.buffer.find(think_end_token) {
                let reasoning_text = self.buffer[..end_idx].to_string();

                self.in_reasoning = false;
                let normal_text = self.buffer[end_idx + think_end_token.len()..].to_string();
                self.buffer.clear();

                return Ok((Some(normal_text), Some(reasoning_text.trim_end().to_string())));
            }
        }

        // Continue with reasoning content
        if self.in_reasoning {
            // Check for tool_start_token interruption
            if let Some(tool_token) = tool_start_token {
                if self.buffer.contains(tool_token) {
                    if let Some(tool_idx) = self.buffer.find(tool_token) {
                        let reasoning_text = self.buffer[..tool_idx].to_string();
                        let normal_text = self.buffer[tool_idx..].to_string();
                        self.buffer.clear();
                        self.in_reasoning = false;
                        return Ok((Some(normal_text), Some(reasoning_text)));
                    }
                }
            }
            if stream_reasoning {
                let reasoning_text = self.buffer.clone();
                self.buffer.clear();
                return Ok((None, Some(reasoning_text)));
            } else {
                return Ok((None, None));
            }
        }

        // If we're not in a reasoning block return as normal text
        if !self.in_reasoning {
            let normal_text = self.buffer.clone();
            self.buffer.clear();
            return Ok((Some(normal_text), None));
        }

        Ok((None, None))
    }
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_class::<RustReasoningState>()?;
    Ok(())
}
