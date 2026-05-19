use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

// Existing code

#[pyfunction]
fn trim_overlap(existing_text: &str, new_chunk: &str) -> String {
    let max_possible = existing_text.len().min(new_chunk.len());
    let mut max_overlap = 0;

    for i in 1..=max_possible {
        if existing_text.ends_with(&new_chunk[..i]) {
            max_overlap = i;
        }
    }

    if max_overlap == new_chunk.len() {
        "".to_string()
    } else {
        new_chunk[max_overlap..].to_string()
    }
}

use ignore::WalkBuilder;

#[pyfunction]
fn find_files(model_path: &str) -> PyResult<Vec<String>> {
    let mut files = Vec::new();

    let walker = WalkBuilder::new(model_path).follow_links(true).build();

    for result in walker {
        match result {
            Ok(entry) => {
                let path = entry.path();
                if path.is_file() {
                    if let Some(path_str) = path.to_str() {
                        files.push(path_str.to_string());
                    }
                }
            }
            Err(err) => return Err(pyo3::exceptions::PyIOError::new_err(err.to_string())),
        }
    }

    Ok(files)
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

    fn detect_and_parse(
        &mut self,
        text: &str,
        think_start_token: &str,
        think_end_token: &str,
        tool_start_token: Option<&str>,
        previous_content: &str,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let in_reasoning = self.in_reasoning || text.contains(think_start_token);

        if !in_reasoning {
            return Ok((Some(text.to_string()), None));
        }

        let processed_text = text.replacen(think_start_token, "", 1);
        let processed_text = processed_text.trim();

        if !processed_text.contains(think_end_token) && !previous_content.contains(think_end_token)
        {
            if in_reasoning {
                if let Some(tool_token) = tool_start_token {
                    if processed_text.contains(tool_token) {
                        if let Some(tool_idx) = processed_text.find(tool_token) {
                            let reasoning_text = processed_text[..tool_idx].trim().to_string();
                            let normal_text = processed_text[tool_idx..].to_string();
                            return Ok((Some(normal_text), Some(reasoning_text)));
                        }
                    }
                }
            }
            // Reasoning text was truncated before end token
            return Ok((None, Some(processed_text.to_string())));
        }

        if processed_text.contains(think_end_token) {
            let mut splits = processed_text.splitn(2, think_end_token);
            let reasoning_text = splits.next().unwrap_or("").to_string();
            let normal_text = splits.next().unwrap_or("").trim_start().to_string();
            return Ok((Some(normal_text), Some(reasoning_text)));
        }

        Ok((Some(processed_text.to_string()), None))
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
        let current_text = self.buffer.clone();

        let mut tokens_to_check = vec![think_start_token, think_end_token];
        if let Some(tool_start) = tool_start_token {
            tokens_to_check.push(tool_start);
        }

        if tokens_to_check
            .iter()
            .any(|&token| token.starts_with(&current_text) && token != current_text)
        {
            return Ok((None, None));
        }

        let mut current_text = current_text;
        if !self.stripped_think_start && current_text.contains(think_start_token) {
            current_text = current_text.replace(think_start_token, "");
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }

        if self.in_reasoning && current_text.contains(think_end_token) {
            if let Some(end_idx) = current_text.find(think_end_token) {
                let reasoning_text = current_text[..end_idx].to_string();
                self.buffer.clear();
                self.in_reasoning = false;
                let normal_text = current_text[end_idx + think_end_token.len()..].to_string();
                return Ok((Some(normal_text), Some(reasoning_text.trim_end().to_string())));
            }
        }

        if self.in_reasoning {
            if let Some(tool_start) = tool_start_token {
                if current_text.contains(tool_start) {
                    if let Some(tool_idx) = current_text.find(tool_start) {
                        let reasoning_text = current_text[..tool_idx].to_string();
                        let normal_text = current_text[tool_idx..].to_string();
                        self.buffer.clear();
                        self.in_reasoning = false;
                        return Ok((Some(normal_text), Some(reasoning_text)));
                    }
                }
            }

            if stream_reasoning {
                self.buffer.clear();
                return Ok((None, Some(current_text)));
            } else {
                return Ok((None, None));
            }
        }

        if !self.in_reasoning {
            self.buffer.clear();
            return Ok((Some(current_text), None));
        }

        Ok((None, None))
    }
}

#[pyfunction]
#[pyo3(signature = (msg_dict, content_format, image_data, video_data, audio_data, modalities, use_dpsk_v32_encoding))]
fn process_content_for_template_format<'py>(
    py: Python<'py>,
    msg_dict: Bound<'py, PyDict>,
    content_format: &str,
    image_data: Bound<'py, PyList>,
    video_data: Bound<'py, PyList>,
    audio_data: Bound<'py, PyList>,
    modalities: Bound<'py, PyList>,
    use_dpsk_v32_encoding: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let sglang_utils = py.import("sglang.srt.utils")?;
    let image_data_cls = sglang_utils.getattr("ImageData")?;

    let content_obj = msg_dict.get_item("content")?;
    let content_list = match content_obj {
        Some(obj) => {
            if obj.is_instance_of::<PyList>() {
                obj.downcast_into::<PyList>().unwrap()
            } else {
                let new_msg = PyDict::new(py);
                for (k, v) in msg_dict.iter() {
                    if !v.is_none() {
                        new_msg.set_item(k, v)?;
                    }
                }
                return Ok(new_msg);
            }
        }
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
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
    m.add_class::<RustReasoningState>()?;
    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;
    Ok(())
}
