use pyo3::prelude::*;
<<<<<<< HEAD
use pyo3::types::{PyAny, PyDict};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

fn update_hash_with_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn update_hash_with_token(hasher: &mut Sha256, token: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(value) = token.extract::<u32>() {
        update_hash_with_u32(hasher, value);
        return Ok(());
    }

    if let Ok(iter) = token.try_iter() {
        for elem in iter {
            let value = elem?.extract::<u32>().map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "token tuple/list elements must be unsigned 32-bit integers: {err}"
                ))
            })?;
            update_hash_with_u32(hasher, value);
        }
        return Ok(());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "tokens must be unsigned 32-bit integers or iterable bigram tokens",
    ))
}

fn new_chained_hasher(prior_hash: Option<&str>) -> PyResult<Sha256> {
    let mut hasher = Sha256::new();
    if let Some(prior_hash) = prior_hash {
        if !prior_hash.is_empty() {
            let decoded = hex::decode(prior_hash).map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "prior_hash must be a valid hex SHA256 digest: {err}"
                ))
            })?;
            hasher.update(decoded);
        }
    }
    Ok(hasher)
}

fn finalize_hash(hasher: Sha256) -> String {
    hex::encode(hasher.finalize())
}

fn saguaro_prefix_hash_parts(parts: &[String], window: usize) -> String {
    let start = if window == 0 || parts.len() < window {
        0
    } else {
        parts.len() - window
    };
    let raw = parts[start..].join(",");
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    finalize_hash(hasher)
}

#[pyfunction]
#[pyo3(signature = (token_ids, prior_hash = None))]
fn hicache_hash(token_ids: &Bound<'_, PyAny>, prior_hash: Option<&str>) -> PyResult<String> {
    let mut hasher = new_chained_hasher(prior_hash)?;
    let iter = token_ids.try_iter().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("token_ids must be an iterable of tokens")
    })?;
    for token in iter {
        update_hash_with_token(&mut hasher, &token?)?;
    }
    Ok(finalize_hash(hasher))
}

#[pyfunction]
#[pyo3(signature = (token_ids, page_size, prior_hash = None))]
fn hicache_page_hashes(
    token_ids: &Bound<'_, PyAny>,
    page_size: usize,
    prior_hash: Option<&str>,
) -> PyResult<Vec<String>> {
    if page_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "page_size must be greater than zero",
        ));
    }

    let iter = token_ids.try_iter().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("token_ids must be an iterable of tokens")
    })?;
    let mut hashes = Vec::new();
    let mut parent_hash = prior_hash.map(str::to_owned);
    let mut hasher = new_chained_hasher(parent_hash.as_deref())?;
    let mut page_tokens = 0_usize;

    for token in iter {
        update_hash_with_token(&mut hasher, &token?)?;
        page_tokens += 1;
        if page_tokens == page_size {
            let hash = finalize_hash(hasher);
            parent_hash = Some(hash.clone());
            hashes.push(hash);
            hasher = new_chained_hasher(parent_hash.as_deref())?;
            page_tokens = 0;
        }
    }

    if page_tokens > 0 {
        hashes.push(finalize_hash(hasher));
    }

    Ok(hashes)
}

#[pyfunction]
fn hicache_hash_to_int64(hash_str: &str) -> PyResult<i64> {
    if hash_str.len() < 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hash_str must contain at least 16 hex characters",
        ));
    }
    let value = u64::from_str_radix(&hash_str[..16], 16).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "hash_str must start with 16 valid hex characters: {err}"
        ))
    })?;
    Ok(value as i64)
}

#[pyfunction]
fn saguaro_prefix_hash(tokens: &Bound<'_, PyAny>, window: usize) -> PyResult<String> {
    let iter = tokens
        .try_iter()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("tokens must be an iterable"))?;
    let mut parts = Vec::new();
    for token in iter {
        let token = token?;
        if let Ok(value) = token.extract::<i64>() {
            parts.push(value.to_string());
        } else {
            parts.push(token.str()?.to_string_lossy().to_string());
        }
    }
    Ok(saguaro_prefix_hash_parts(&parts, window))
}

#[pyfunction]
fn pack_sampling_params(
    reqs: &Bound<'_, PyAny>,
    top_k_all: i32,
    enable_deterministic: bool,
    enable_custom_logit_processor: bool,
) -> PyResult<(
    Vec<f32>,
    Vec<f32>,
    Vec<i32>,
    Vec<f32>,
    Option<Vec<i64>>,
    bool,
    bool,
    bool,
    bool,
    bool,
)> {
    let iter = reqs
        .try_iter()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("reqs must be iterable"))?;

    let mut temperatures = Vec::new();
    let mut top_ps = Vec::new();
    let mut top_ks = Vec::new();
    let mut min_ps = Vec::new();
    let mut sampling_seed = if enable_deterministic {
        Some(Vec::new())
    } else {
        None
    };

    let mut is_all_greedy = true;
    let mut need_top_p_sampling = false;
    let mut need_top_k_sampling = false;
    let mut need_min_p_sampling = false;
    let mut has_custom_logit_processor = false;

    for req in iter {
        let req = req?;
        let params = req.getattr("sampling_params")?;

        let temperature = params.getattr("temperature")?.extract::<f32>()?;
        let top_p = params.getattr("top_p")?.extract::<f32>()?;
        let top_k = params.getattr("top_k")?.extract::<i32>()?;
        let min_p = params.getattr("min_p")?.extract::<f32>()?;

        temperatures.push(temperature);
        top_ps.push(top_p);
        top_ks.push(top_k);
        min_ps.push(min_p);

        is_all_greedy &= top_k <= 1;
        need_top_p_sampling |= top_p != 1.0;
        need_top_k_sampling |= top_k != top_k_all;
        need_min_p_sampling |= min_p > 0.0;

        if let Some(seeds) = sampling_seed.as_mut() {
            let seed = params.getattr("sampling_seed")?;
            if seed.is_none() {
                seeds.push(42);
            } else {
                seeds.push(seed.extract::<i64>()?);
            }
        }

        if enable_custom_logit_processor && req.getattr("custom_logit_processor")?.is_truthy()? {
            has_custom_logit_processor = true;
        }
    }

    Ok((
        temperatures,
        top_ps,
        top_ks,
        min_ps,
        sampling_seed,
        is_all_greedy,
        need_top_p_sampling,
        need_top_k_sampling,
        need_min_p_sampling,
        has_custom_logit_processor,
    ))
}
=======
use pyo3::types::{PyDict, PyList, PyString};

// Existing code
>>>>>>> e0905f017488c752faeed9aedcf62d0ec397d020

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

#[pyfunction]
fn find_files(path: &str) -> PyResult<Vec<String>> {
    let mut files = Vec::new();
    let walker = ignore::WalkBuilder::new(path)
        .follow_links(true)
        .hidden(false)
        .git_ignore(false)
        .build();

    for result in walker {
        let entry = result.map_err(|err| pyo3::exceptions::PyIOError::new_err(err.to_string()))?;
        if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            if let Some(path) = entry.path().to_str() {
                files.push(path.to_string());
            }
        }
    }

    files.sort();
    Ok(files)
}

fn compute_sha256(path: &Path) -> PyResult<(String, u64)> {
    let file = File::open(path).map_err(|err| {
        pyo3::exceptions::PyIOError::new_err(format!("{}: {}", path.display(), err))
    })?;
    let size = file
        .metadata()
        .map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!("{}: {}", path.display(), err))
        })?
        .len();
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let bytes = reader.read(&mut buffer).map_err(|err| {
            pyo3::exceptions::PyIOError::new_err(format!("{}: {}", path.display(), err))
        })?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }

    Ok((hex::encode(hasher.finalize()), size))
}

#[pyfunction]
#[pyo3(signature = (model_path, filenames, max_workers = 4))]
fn sha256_manifest<'py>(
    py: Python<'py>,
    model_path: &str,
    filenames: Vec<String>,
    max_workers: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let model_path = PathBuf::from(model_path);
    let workers = max_workers.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;

    let results: PyResult<Vec<(String, Option<(String, u64)>)>> = pool.install(|| {
        filenames
            .par_iter()
            .map(|filename| {
                let full_path = model_path.join(filename);
                if !full_path.exists() {
                    return Ok((filename.clone(), None));
                }
                compute_sha256(&full_path).map(|info| (filename.clone(), Some(info)))
            })
            .collect()
    });

    let manifest = PyDict::new(py);
    for (filename, info) in results? {
        if let Some((sha256, size)) = info {
            manifest.set_item(filename, (sha256, size))?;
        }
    }
    Ok(manifest)
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
<<<<<<< HEAD
    m.add_function(wrap_pyfunction!(sha256_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_page_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash_to_int64, m)?)?;
    m.add_function(wrap_pyfunction!(saguaro_prefix_hash, m)?)?;
    m.add_function(wrap_pyfunction!(pack_sampling_params, m)?)?;
=======
    m.add_class::<RustReasoningState>()?;
    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;
>>>>>>> e0905f017488c752faeed9aedcf62d0ec397d020
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sglang_rust_utils_test_{suffix}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn trims_overlap_on_utf8_boundary() {
        assert_eq!(trim_overlap("hello wor", "world"), "ld");
        assert_eq!(trim_overlap("prefix café", "café next"), " next");
    }

    #[test]
    fn discovers_files() {
        let dir = temp_dir();
        fs::write(dir.join("a.safetensors"), b"a").unwrap();
        fs::create_dir_all(dir.join("nested")).unwrap();
        fs::write(dir.join("nested").join("b.bin"), b"b").unwrap();

        let files = find_files(dir.to_str().unwrap()).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p.ends_with("a.safetensors")));
        assert!(files.iter().any(|p| p.ends_with("nested/b.bin")));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn computes_sha256_and_size() {
        let dir = temp_dir();
        let mut file = File::create(dir.join("weights.bin")).unwrap();
        file.write_all(b"abc").unwrap();

        let (sha256, size) = compute_sha256(&dir.join("weights.bin")).unwrap();
        assert_eq!(
            sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(size, 3);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn converts_hash_prefix_to_signed_i64() {
        assert_eq!(
            hicache_hash_to_int64("7fffffffffffffff0000").unwrap(),
            i64::MAX
        );
        assert_eq!(
            hicache_hash_to_int64("80000000000000000000").unwrap(),
            i64::MIN
        );
        assert_eq!(hicache_hash_to_int64("ffffffffffffffff0000").unwrap(), -1);
    }

    #[test]
    fn hashes_saguaro_prefix_window() {
        let parts = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let expected = hex::encode(Sha256::digest(b"2,3"));
        assert_eq!(saguaro_prefix_hash_parts(&parts, 2), expected);

        let expected_all = hex::encode(Sha256::digest(b"1,2,3"));
        assert_eq!(saguaro_prefix_hash_parts(&parts, 0), expected_all);
    }
}
