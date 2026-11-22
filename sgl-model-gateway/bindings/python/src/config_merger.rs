use pyo3::prelude::*;
use serde_yaml::Value;
use std::collections::HashMap;
use std::fs;

#[pyclass]
pub struct ConfigArgumentMerger {
    store_true_actions: Vec<String>,
    unsupported_actions: HashMap<String, String>,
}

#[pymethods]
impl ConfigArgumentMerger {
    #[new]
    #[pyo3(signature = (store_true_actions=None, unsupported_actions=None))]
    fn new(
        store_true_actions: Option<Vec<String>>,
        unsupported_actions: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            store_true_actions: store_true_actions.unwrap_or_default(),
            unsupported_actions: unsupported_actions.unwrap_or_default(),
        }
    }

    fn merge_config_with_args(&self, cli_args: Vec<String>) -> PyResult<Vec<String>> {
        let mut config_index = None;
        for (i, arg) in cli_args.iter().enumerate() {
            if arg == "--config" {
                if config_index.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Multiple config files specified! Only one allowed.",
                    ));
                }
                config_index = Some(i);
            }
        }

        let config_index = match config_index {
            Some(idx) => idx,
            None => return Ok(cli_args),
        };

        if config_index == cli_args.len() - 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No config file specified after --config flag!",
            ));
        }

        let config_file_path = &cli_args[config_index + 1];

        let path = std::path::Path::new(config_file_path);

        if !path.exists() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Config file not found: {}", config_file_path
            )));
        }

        let contents = fs::read_to_string(config_file_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to read config file {}: {}", config_file_path, e
            ))
        })?;

        let config_data: serde_yaml::Mapping = if contents.trim().is_empty() {
            serde_yaml::Mapping::new()
        } else {
            serde_yaml::from_str(&contents).map_err(|_e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Config file must contain a dictionary at root level"
                ))
            })?
        };

        let mut config_args = Vec::new();
        for (key, value) in config_data.iter() {
            let key_str = match key.as_str() {
                Some(s) => s,
                None => continue,
            };

            let key_norm = key_str.replace("-", "_");
            if let Some(action) = self.unsupported_actions.get(&key_norm) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported config option '{}' with action '{}'",
                    key_norm, action
                )));
            }

            match value {
                Value::Bool(b) => {
                    if self.store_true_actions.contains(&key_norm) {
                        if *b {
                            config_args.push(format!("--{}", key_str));
                        }
                    } else {
                        config_args.push(format!("--{}", key_str));
                        config_args.push(b.to_string());
                    }
                }
                Value::Sequence(seq) => {
                    if !seq.is_empty() {
                        config_args.push(format!("--{}", key_str));
                        for item in seq {
                            let item_str = match item {
                                Value::String(s) => s.clone(),
                                Value::Number(n) => n.to_string(),
                                Value::Bool(b) => b.to_string(),
                                _ => continue, // Skip complex types
                            };
                            config_args.push(item_str);
                        }
                    }
                }
                Value::String(s) => {
                    config_args.push(format!("--{}", key_str));
                    config_args.push(s.clone());
                }
                Value::Number(n) => {
                    config_args.push(format!("--{}", key_str));
                    config_args.push(n.to_string());
                }
                _ => continue, // Skip null and mapping
            }
        }

        let mut result = config_args;
        result.extend_from_slice(&cli_args[..config_index]);
        result.extend_from_slice(&cli_args[config_index + 2..]);

        Ok(result)
    }
}
