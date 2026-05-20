use pyo3::prelude::*;
use std::collections::HashMap;

// User provided this code:
#[pyclass]
#[derive(Debug)]
pub struct TreeNode {
    pub key: String,
    pub value: Option<PyObject>,           // or whatever Python stores
    pub children: HashMap<String, TreeNode>,
    // ... other fields (is_leaf, etc.)
}

#[pymethods]
impl TreeNode {
    // port _insert_helper logic here
}
