use anyhow::Result;
use ignore::{WalkBuilder, WalkState};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileInfo {
    pub sha256: String,
    pub size: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Manifest {
    pub files: BTreeMap<String, FileInfo>,
}

pub fn generate_checksums(source_dir: &Path, max_workers: usize, explicit_files: Option<Vec<String>>) -> Result<Manifest> {
    let files: Vec<PathBuf> = if let Some(filenames) = explicit_files {
        filenames.into_iter().map(|f| source_dir.join(f)).collect()
    } else {
        let ignore_patterns = vec![
            ".DS_Store",
            "*.lock",
            ".gitattributes",
            "LICENSE",
            "LICENSE.*",
            "README.md",
            "README.*",
            "NOTICE",
        ];

        let mut builder = WalkBuilder::new(source_dir);
        builder.hidden(false).parents(false).ignore(false).git_ignore(false).git_global(false).git_exclude(false);
        builder.follow_links(true);

        let mut overrides = ignore::overrides::OverrideBuilder::new(source_dir);
        for pattern in ignore_patterns {
            overrides.add(&format!("!{}", pattern)).unwrap();
        }
        let overrides = overrides.build().unwrap();
        builder.overrides(overrides);

        let (tx, rx) = std::sync::mpsc::channel();

        builder.build_parallel().run(|| {
            let tx = tx.clone();
            Box::new(move |result| {
                if let Ok(entry) = result {
                    if let Some(ft) = entry.file_type() {
                        if ft.is_file() {
                            let path = entry.path();
                            let name = path.file_name().unwrap_or_default().to_string_lossy();
                            if !name.starts_with('.') {
                                tx.send(path.to_path_buf()).unwrap();
                            }
                        }
                    }
                }
                WalkState::Continue
            })
        });

        drop(tx);
        rx.into_iter().collect()
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(max_workers).build()?;

    let results: Result<Vec<Option<(String, FileInfo)>>> = pool.install(|| {
        files.into_par_iter()
            .map(|path| {
                let rel_path = path.strip_prefix(source_dir).unwrap_or(&path);
                let name = rel_path.to_string_lossy().replace("\\", "/");

                if !path.exists() || !path.is_file() {
                    return Ok(None);
                }

                let mut file = File::open(&path)?;
                let size = file.metadata()?.len() as i64;

                let mut hasher = Sha256::new();
                let mut buffer = [0; 128 * 1024];

                loop {
                    let count = file.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }

                let sha256 = hex::encode(hasher.finalize());

                Ok(Some((name, FileInfo { sha256, size })))
            })
            .collect()
    });

    let mut files_map = BTreeMap::new();
    for info_opt in results? {
        if let Some((name, info)) = info_opt {
            files_map.insert(name, info);
        }
    }

    Ok(Manifest { files: files_map })
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (source, max_workers=4))]
fn generate_checksums_py(source: String, max_workers: usize) -> PyResult<String> {
    let path = Path::new(&source);
    let manifest = generate_checksums(path, max_workers, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    serde_json::to_string(&manifest)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn verify_checksums_py(model_path: String, manifest_json: String, max_workers: usize) -> PyResult<Vec<String>> {
    let manifest: Manifest = serde_json::from_str(&manifest_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let path = Path::new(&model_path);
    let explicit_files: Vec<String> = manifest.files.keys().cloned().collect();

    let actual_manifest = generate_checksums(path, max_workers, Some(explicit_files))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut errors = Vec::new();

    for (filename, exp) in manifest.files {
        if let Some(act) = actual_manifest.files.get(&filename) {
            if act.sha256 != exp.sha256 {
                errors.push(format!("{}: mismatch (expected={}... size={}, actual={}... size={})",
                    filename, &exp.sha256[..16.min(exp.sha256.len())], exp.size,
                    &act.sha256[..16.min(act.sha256.len())], act.size));
            }
        } else {
            errors.push(format!("{}: missing (expected size={})", filename, exp.size));
        }
    }

    Ok(errors)
}

#[cfg(feature = "python")]
pub fn add_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_checksums_py, m)?)?;
    m.add_function(wrap_pyfunction!(verify_checksums_py, m)?)?;
    Ok(())
}
