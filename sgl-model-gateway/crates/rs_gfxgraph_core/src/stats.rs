use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GfxGraphStatsSample {
    pub samples: u64,
    pub failures: u64,
}

impl GfxGraphStatsSample {
    pub fn new(samples: u64, failures: u64) -> Self {
        Self { samples, failures }
    }
}
