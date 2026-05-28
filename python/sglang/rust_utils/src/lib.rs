use pyo3::prelude::*;

use jsonschema::validator_for;
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

static ITERATION_RE: OnceLock<Regex> = OnceLock::new();

#[pyfunction]
fn detect_jinja_template_content_format(chat_template: &str) -> PyResult<String> {
    // Check for multimodal keywords
    if chat_template.contains("image")
        || chat_template.contains("audio")
        || chat_template.contains("video")
        || chat_template.contains("vision")
    {
        return Ok("openai".to_string());
    }

    // Try to find content iteration loop
    let iteration_re = ITERATION_RE.get_or_init(|| {
        Regex::new(r"\{%[-]?\s*for\s+[a-zA-Z0-9_]+\s+in\s+(?:message\['content'\]|msg\.content|m\.content)\s*[-]?%\}").unwrap()
    });

    if iteration_re.is_match(chat_template) {
        return Ok("openai".to_string());
    }

    Ok("string".to_string())
}

use pyo3::types::{PyAny, PyDict, PyList};
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

#[pyfunction]
fn trim_overlap(existing_text: &str, new_chunk: &str) -> String {
    let max_possible = existing_text.len().min(new_chunk.len());
    let mut max_overlap = 0;

    for i in 1..=max_possible {
        if new_chunk.is_char_boundary(i) && existing_text.ends_with(&new_chunk[..i]) {
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

fn is_chinese_char(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

#[pyfunction]
fn find_printable_text(text: &str) -> String {
    if text.ends_with('\n') {
        return text.to_string();
    }

    if text.is_empty() {
        return String::new();
    }

    if text
        .chars()
        .last()
        .is_some_and(|last_char| is_chinese_char(last_char as u32))
    {
        return text.to_string();
    }

    let mut rev_indices = text.char_indices().rev();
    let _last = rev_indices.next();
    if let Some((idx, penultimate_char)) = rev_indices.next() {
        if is_chinese_char(penultimate_char as u32) {
            return text[..idx + penultimate_char.len_utf8()].to_string();
        }
    }

    if let Some(space_idx) = text.rfind(' ') {
        return text[..=space_idx].to_string();
    }

    String::new()
}

fn compile_jsonschema_value(schema: Value) -> PyResult<()> {
    validator_for(&schema).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid schema definition: {err}"))
    })?;
    Ok(())
}

#[pyfunction]
fn check_jsonschema(schema_str: &str) -> PyResult<bool> {
    let schema = serde_json::from_str::<Value>(schema_str).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid schema JSON: {err}"))
    })?;
    compile_jsonschema_value(schema)?;
    Ok(true)
}

#[pyfunction]
fn check_schema_fast(schema: &Bound<'_, PyAny>) -> PyResult<()> {
    let schema_json = schema
        .py()
        .import("json")?
        .getattr("dumps")?
        .call1((schema,))?
        .extract::<String>()?;
    let schema = serde_json::from_str::<Value>(&schema_json).map_err(|err| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid schema JSON: {err}"))
    })?;
    compile_jsonschema_value(schema)
}

#[pyfunction]
fn find_common_prefix(s1: &str, s2: &str) -> String {
    let mut prefix_len = 0;
    for (b1, b2) in s1.bytes().zip(s2.bytes()) {
        if b1 == b2 {
            prefix_len += 1;
        } else {
            break;
        }
    }

    // Ensure we don't slice inside a character boundary
    while prefix_len > 0 && !s1.is_char_boundary(prefix_len) {
        prefix_len -= 1;
    }

    s1[..prefix_len].to_string()
}

#[pymodule]
fn sglang_rust_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_printable_text, m)?)?;
    m.add_function(wrap_pyfunction!(find_common_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(trim_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_page_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash_to_int64, m)?)?;
    m.add_function(wrap_pyfunction!(saguaro_prefix_hash, m)?)?;
    m.add_function(wrap_pyfunction!(pack_sampling_params, m)?)?;
    m.add_class::<RustReasoningState>()?;
    m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;
    m.add_function(wrap_pyfunction!(detect_jinja_template_content_format, m)?)?;
    m.add_function(wrap_pyfunction!(check_jsonschema, m)?)?;
    m.add_function(wrap_pyfunction!(check_schema_fast, m)?)?;
    m.add_function(wrap_pyfunction!(parse_message_from_completion_text_dsv32, m)?)?;
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

    #[test]
    fn computes_common_prefix() {
        assert_eq!(find_common_prefix("hello world", "hello friend"), "hello ");
        assert_eq!(find_common_prefix("abc", "xyz"), "");
        assert_eq!(find_common_prefix("café next", "café prev"), "café ");
    }
}

// =============== DSV32 Parsing ================

fn _read_until_stop<'a>(index: usize, text: &'a str, stops: &[&str]) -> (usize, &'a str, Option<String>) {
    let mut min_pos = text.len();
    let mut matched_stop = None;

    for &s in stops {
        if let Some(pos) = text[index..].find(s) {
            let actual_pos = index + pos;
            if actual_pos < min_pos {
                min_pos = actual_pos;
                matched_stop = Some(s);
            }
        }
    }

    if let Some(stop) = matched_stop {
        let content = &text[index..min_pos];
        (min_pos + stop.len(), content, Some(stop.to_string()))
    } else {
        let content = &text[index..];
        (text.len(), content, None)
    }
}

use std::collections::HashMap;

#[pyfunction]
#[pyo3(text_signature = "(text, thinking_mode)")]
fn parse_message_from_completion_text_dsv32<'py>(
    py: Python<'py>,
    text: &str,
    thinking_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let dsml_token = "｜DSML｜";
    let thinking_end_token = "</think>";
    let eos_token = "<｜end of sentence｜>";
    let thinking_start_token = "<think>";
    let bos_token = "<｜begin of sentence｜>";

    let mut index = 0;
    let mut stop_token: Option<String> = None;
    let tool_calls_start_token = format!("\n\n<{}function_calls", dsml_token);

    let is_thinking = thinking_mode == "thinking";
    let mut is_tool_calling = false;
    let mut reasoning_content = "";

    if is_thinking {
        let (new_idx, content_delta, st) = _read_until_stop(index, text, &[thinking_end_token, &tool_calls_start_token]);
        index = new_idx;
        reasoning_content = content_delta;
        stop_token = st.clone();
        if stop_token.as_deref() != Some(thinking_end_token) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid thinking format"));
        }
    }

    let (mut new_idx, mut summary_content, mut st) = _read_until_stop(index, text, &[eos_token, &tool_calls_start_token]);
    index = new_idx;
    stop_token = st.clone();

    if stop_token.as_deref() == Some(&tool_calls_start_token) {
        is_tool_calling = true;
    } else if stop_token.as_deref() != Some(eos_token) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid summary format"));
    }

    let mut tool_calls = Vec::new();

    if is_tool_calling {
        let tool_calls_end_token = format!("</{}function_calls>", dsml_token);

        while index < text.len() {
            let invoke_start = format!("<{}invoke", dsml_token);
            let (new_idx, delta, st) = _read_until_stop(index, text, &[&invoke_start, &tool_calls_end_token]);
            index = new_idx;

            if delta != ">\n" {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Tool call format error"));
            }
            if st.as_deref() == Some(&tool_calls_end_token) {
                stop_token = st;
                break;
            }
            if st.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing special token"));
            }

            let param_start = format!("<{}parameter", dsml_token);
            let invoke_end = format!("</{}invoke", dsml_token);
            let (new_idx, tool_name_content, st) = _read_until_stop(index, text, &[&param_start, &invoke_end]);
            index = new_idx;


            let tool_name = if let Some(start) = tool_name_content.find("name=\"") {
                if let Some(end) = tool_name_content[start + 6..].find("\">\n") {
                    tool_name_content[start + 6..start + 6 + end].to_string()
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Tool name format error"));
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Tool name format error"));
            };


            let mut tool_args: HashMap<String, (String, String)> = HashMap::new();
            let mut current_st = st;

            while current_st.as_deref() == Some(&param_start) {
                let param_end = format!("/{}parameter", dsml_token);
                let (new_idx, param_content, st2) = _read_until_stop(index, text, &[&param_end]);
                index = new_idx;


                let mut valid = false;
                if let Some(name_start) = param_content.find("name=\"") {
                    if let Some(name_end) = param_content[name_start + 6..].find("\" string=\"") {
                        let name_val = &param_content[name_start + 6..name_start + 6 + name_end];
                        let string_start = name_start + 6 + name_end + 10;
                        if let Some(string_end) = param_content[string_start..].find("\">") {
                            let str_val = &param_content[string_start..string_start + string_end];
                            let content_start = string_start + string_end + 2;
                            if let Some(content_end) = param_content[content_start..].rfind("<") {
                                let content_val = &param_content[content_start..content_start + content_end];

                                if tool_args.contains_key(name_val) {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Duplicate parameter name"));
                                }
                                tool_args.insert(name_val.to_string(), (content_val.to_string(), str_val.to_string()));
                                valid = true;
                            }
                        }
                    }
                }
                if !valid {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Parameter format error"));
                }


                let (new_idx, content2, st3) = _read_until_stop(index, text, &[&param_start, &invoke_end]);
                index = new_idx;
                if content2 != ">\n" {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Parameter format error"));
                }
                current_st = st3;
            }




            let mut args_map = serde_json::Map::new();
            for (k, (v, s)) in tool_args.iter() {
                if s == "true" {
                    args_map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
                } else {
                    match serde_json::from_str::<serde_json::Value>(v) {
                        Ok(json_val) => {
                            args_map.insert(k.to_string(), json_val);
                        }
                        Err(_) => {
                            // Fallback if it fails to parse as valid JSON (shouldn't happen with correct inputs)
                            args_map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
                        }
                    }
                }
            }
            let args_json = serde_json::to_string(&serde_json::Value::Object(args_map)).unwrap();

            let tool_func = PyDict::new(py);
            tool_func.set_item("name", tool_name)?;
            tool_func.set_item("arguments", args_json)?;

            let tool_call = PyDict::new(py);
            tool_call.set_item("type", "function")?;
            tool_call.set_item("function", tool_func)?;
            tool_calls.push(tool_call);
        }

        let (new_idx, tool_ends_text, st) = _read_until_stop(index, text, &[eos_token]);
        index = new_idx;
        stop_token = st;
        if !tool_ends_text.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unexpected content after tool calls"));
        }
    }

    if index != text.len() && stop_token.as_deref() != Some(eos_token) && stop_token.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unexpected content at end"));
    }

    for sp_token in &[bos_token, eos_token, thinking_start_token, thinking_end_token, dsml_token] {
        if summary_content.contains(sp_token) || reasoning_content.contains(sp_token) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unexpected special token in content"));
        }
    }

    let result = PyDict::new(py);
    result.set_item("role", "assistant")?;
    result.set_item("content", summary_content)?;
    result.set_item("reasoning_content", reasoning_content)?;

    let py_tool_calls = PyList::new(py, tool_calls)?;
    result.set_item("tool_calls", py_tool_calls)?;

    Ok(result)
}
