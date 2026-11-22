use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GfxGraphNodeSpec {
    name: String,
    impl_path: String,
}

impl GfxGraphNodeSpec {
    pub fn new(name: impl Into<String>, impl_path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            impl_path: impl_path.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn impl_path(&self) -> &str {
        &self.impl_path
    }
}
