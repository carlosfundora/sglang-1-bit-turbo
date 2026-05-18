use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

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
}
