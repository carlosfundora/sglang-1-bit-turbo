use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// Existing code

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


#[pyfunction]
fn process_content_for_template_format<'py>(
    py: Python<'py>,
    msg_dict: &Bound<'py, PyDict>,
    content_format: &str,
    image_data: &Bound<'py, PyList>,
    video_data: &Bound<'py, PyList>,
    audio_data: &Bound<'py, PyList>,
    modalities: &Bound<'py, PyList>,
    use_dpsk_v32_encoding: bool,
    image_data_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let content_opt = msg_dict.get_item("content")?;

    // If not list, return dict without None values
    let content_list = match content_opt {
        Some(c) => {
            if c.is_instance_of::<PyList>() {
                c.downcast_into::<PyList>().unwrap()
            } else {
                let new_msg = PyDict::new(py);
                for (k, v) in msg_dict.iter() {
                    if !v.is_none() {
                        new_msg.set_item(k, v)?;
                    }
                }
                return Ok(new_msg);
            }
        },
        None => {
            let new_msg = PyDict::new(py);
            for (k, v) in msg_dict.iter() {
                if !v.is_none() {
                    new_msg.set_item(k, v)?;
                }
            }
            return Ok(new_msg);
        }
    };

    if content_format == "openai" || use_dpsk_v32_encoding {
        let processed_content_parts = PyList::empty(py);
        let mut text_parts = Vec::new();

        for chunk_item in content_list.iter() {
            if let Ok(chunk) = chunk_item.downcast_into::<PyDict>() {
                if let Some(chunk_type_obj) = chunk.get_item("type")? {
                    if let Ok(chunk_type) = chunk_type_obj.extract::<String>() {
                        match chunk_type.as_str() {
                            "image_url" => {
                                let image_obj = chunk.get_item("image_url")?;
                                let image_dict = if let Some(obj) = image_obj {
                                    if obj.is_instance_of::<PyDict>() {
                                        Some(obj.downcast_into::<PyDict>().unwrap())
                                    } else {
                                        None
                                    }
                                } else { None };

                                let mut url = String::new();
                                let mut detail = "auto".to_string();
                                let mut max_dynamic_patch: Option<Py<PyAny>> = None;

                                if let Some(dict) = image_dict {
                                    if let Some(u) = dict.get_item("url")? {
                                        if let Ok(u_str) = u.extract::<String>() {
                                            url = u_str;
                                        }
                                    }
                                    if let Some(d) = dict.get_item("detail")? {
                                        if let Ok(d_str) = d.extract::<String>() {
                                            detail = d_str;
                                        }
                                    }
                                    if let Some(mdp) = dict.get_item("max_dynamic_patch")? {
                                        max_dynamic_patch = Some(mdp.clone().unbind());
                                    }
                                }

                                let kwargs = PyDict::new(py);
                                kwargs.set_item("url", url)?;
                                kwargs.set_item("detail", detail)?;
                                if let Some(mdp) = max_dynamic_patch {
                                    kwargs.set_item("max_dynamic_patch", mdp)?;
                                } else {
                                    kwargs.set_item("max_dynamic_patch", py.None())?;
                                }

                                let image_data_instance = image_data_cls.call((), Some(&kwargs))?;
                                image_data.append(image_data_instance)?;

                                if let Some(mod_val) = chunk.get_item("modalities")? {
                                    modalities.append(mod_val)?;
                                }

                                let new_part = PyDict::new(py);
                                new_part.set_item("type", "image")?;
                                processed_content_parts.append(new_part)?;
                            },
                            "video_url" => {
                                let video_obj = chunk.get_item("video_url")?;
                                let video_dict = if let Some(obj) = video_obj {
                                    if obj.is_instance_of::<PyDict>() {
                                        Some(obj.downcast_into::<PyDict>().unwrap())
                                    } else {
                                        None
                                    }
                                } else { None };

                                let mut url = String::new();
                                let mut max_dynamic_patch: Option<Py<PyAny>> = None;

                                if let Some(dict) = video_dict {
                                    if let Some(u) = dict.get_item("url")? {
                                        if let Ok(u_str) = u.extract::<String>() {
                                            url = u_str;
                                        }
                                    }
                                    if let Some(mdp) = dict.get_item("max_dynamic_patch")? {
                                        max_dynamic_patch = Some(mdp.clone().unbind());
                                    }
                                }

                                if let Some(mdp) = max_dynamic_patch {
                                    let dict_item = PyDict::new(py);
                                    dict_item.set_item("url", url)?;
                                    dict_item.set_item("max_dynamic_patch", mdp)?;
                                    video_data.append(dict_item)?;
                                } else {
                                    video_data.append(url)?;
                                }

                                if let Some(mod_val) = chunk.get_item("modalities")? {
                                    modalities.append(mod_val)?;
                                }

                                let new_part = PyDict::new(py);
                                new_part.set_item("type", "video")?;
                                processed_content_parts.append(new_part)?;
                            },
                            "audio_url" => {
                                let mut url = String::new();
                                let audio_obj = chunk.get_item("audio_url")?;
                                if let Some(obj) = audio_obj {
                                    if let Ok(dict) = obj.downcast_into::<PyDict>() {
                                        if let Some(u) = dict.get_item("url")? {
                                            if let Ok(u_str) = u.extract::<String>() {
                                                url = u_str;
                                            }
                                        }
                                    }
                                }

                                if !url.is_empty() {
                                    audio_data.append(url)?;
                                } else {
                                    // Mirror python gracefully appending an empty string fallback
                                    audio_data.append("")?;
                                }

                                let new_part = PyDict::new(py);
                                new_part.set_item("type", "audio")?;
                                processed_content_parts.append(new_part)?;
                            },
                            "text" => {
                                if use_dpsk_v32_encoding {
                                    if let Some(text_val) = chunk.get_item("text")? {
                                        if let Ok(text_str) = text_val.extract::<String>() {
                                            text_parts.push(text_str);
                                        }
                                    }
                                } else {
                                    processed_content_parts.append(chunk)?;
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }

        let new_msg = PyDict::new(py);
        for (k, v) in msg_dict.iter() {
            if let Ok(k_str) = k.extract::<String>() {
                if k_str != "content" && !v.is_none() {
                    new_msg.set_item(k, v)?;
                }
            }
        }

        if use_dpsk_v32_encoding {
            if text_parts.is_empty() {
                new_msg.set_item("content", "")?;
            } else {
                new_msg.set_item("content", text_parts.join(" "))?;
            }
        } else {
            new_msg.set_item("content", processed_content_parts)?;
        }

        return Ok(new_msg);

    } else if content_format == "string" {
        let mut text_parts = Vec::new();

        for chunk_item in content_list.iter() {
            if let Ok(chunk) = chunk_item.downcast_into::<PyDict>() {
                if let Some(chunk_type_obj) = chunk.get_item("type")? {
                    if let Ok(chunk_type) = chunk_type_obj.extract::<String>() {
                        if chunk_type == "text" {
                            if let Some(text_val) = chunk.get_item("text")? {
                                if let Ok(text_str) = text_val.extract::<String>() {
                                    text_parts.push(text_str);
                                }
                            }
                        }
                    }
                }
            }
        }

        let new_msg = PyDict::new(py);
        for (k, v) in msg_dict.iter() {
            if !v.is_none() {
                new_msg.set_item(k, v)?;
            }
        }

        if text_parts.is_empty() {
            new_msg.set_item("content", "")?;
        } else {
            new_msg.set_item("content", text_parts.join(" "))?;
        }

        return Ok(new_msg);
    }

    Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid content format: {}", content_format)))
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_class::<RustReasoningState>()?;
    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;
    Ok(())
}
