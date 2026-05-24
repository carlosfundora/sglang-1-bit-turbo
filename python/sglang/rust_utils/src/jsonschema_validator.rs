use jsonschema::Validator;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::sync::Arc;

#[pyclass]
pub struct FastJSONSchemaValidator {
    schema: Arc<Validator>,
}

#[pymethods]
impl FastJSONSchemaValidator {
    #[new]
    pub fn new(schema_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let schema_str = schema_dict
            .py()
            .import("json")?
            .getattr("dumps")?
            .call1((schema_dict,))?
            .extract::<String>()?;

        let schema_val: Value = serde_json::from_str(&schema_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON schema: {}", e))
        })?;

        let schema = Validator::new(&schema_val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Schema compilation error: {}", e))
        })?;

        Ok(Self {
            schema: Arc::new(schema),
        })
    }

    pub fn check_schema(&self, instance_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let instance_str = instance_dict
            .py()
            .import("json")?
            .getattr("dumps")?
            .call1((instance_dict,))?
            .extract::<String>()?;

        let instance_val: Value = serde_json::from_str(&instance_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON instance: {}", e))
        })?;

        let mut msgs: Vec<String> = Vec::new();
        for err in self.schema.iter_errors(&instance_val) {
            msgs.push(err.to_string());
        }

        if !msgs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Validation failed: {}",
                msgs.join(", ")
            )));
        }

        Ok(())
    }
}
