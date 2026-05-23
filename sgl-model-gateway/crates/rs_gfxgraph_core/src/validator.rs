//! Pure-Rust graph validator configuration.
//!
//! Extracted from the PyO3 `BridgedGraphValidator` — this module holds
//! the validation policy. The actual tensor comparison (torch.allclose,
//! model forward pass) stays in the PyO3 binding crate since those
//! operations require GIL access.

use serde::{Deserialize, Serialize};

/// Configuration for the graph output validator.
///
/// When validation is enabled, graph replay outputs are compared against
/// eager-mode outputs to detect divergence (e.g. PyTorch issue #155684).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatorConfig {
    /// Whether validation is currently enabled.
    validation_enabled: bool,
}

impl ValidatorConfig {
    /// Create a new validator configuration.
    pub fn new(validation_enabled: bool) -> Self {
        Self { validation_enabled }
    }

    /// Check whether validation should be performed.
    pub fn should_validate(&self) -> bool {
        self.validation_enabled
    }

    /// Enable validation.
    pub fn enable(&mut self) {
        self.validation_enabled = true;
    }

    /// Disable validation.
    pub fn disable(&mut self) {
        self.validation_enabled = false;
    }
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            validation_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let config = ValidatorConfig::default();
        assert!(!config.should_validate());
    }

    #[test]
    fn enable_disable_toggle() {
        let mut config = ValidatorConfig::new(false);
        assert!(!config.should_validate());

        config.enable();
        assert!(config.should_validate());

        config.disable();
        assert!(!config.should_validate());
    }

    #[test]
    fn created_enabled() {
        let config = ValidatorConfig::new(true);
        assert!(config.should_validate());
    }

    #[test]
    fn serde_roundtrip() {
        let config = ValidatorConfig::new(true);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ValidatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }
}
