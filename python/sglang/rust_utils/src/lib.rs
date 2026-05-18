use pyo3::prelude::*;
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
    m.add_function(wrap_pyfunction!(sha256_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_page_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(hicache_hash_to_int64, m)?)?;
    m.add_function(wrap_pyfunction!(saguaro_prefix_hash, m)?)?;
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
