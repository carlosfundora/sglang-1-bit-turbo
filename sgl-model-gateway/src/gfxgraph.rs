use rs_gfxgraph_core::{
    BucketRouterCore, BucketState, DTypeConversionContract, DTypeKind, GfxGraphNodeSpec,
    GfxGraphStatsSample, GraphCapabilityRegistry, GraphRunnerState, LayoutKind, Shape,
    ShapeLayoutConversionPlan, TensorLayout, ValidatorConfig,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SglangGfxGraphContract {
    pub enabled: bool,
    pub validation_enabled: bool,
    pub buckets: Vec<usize>,
    pub decode_impl_path: String,
    pub prefill_impl_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SglangGfxGraphStatus {
    pub enabled: bool,
    pub validation_enabled: bool,
    pub buckets: Vec<usize>,
    pub decode_node: String,
    pub prefill_node: String,
    pub routed_bucket: Option<usize>,
    pub routed_bucket_state: Option<BucketState>,
    pub registry_capabilities: usize,
    pub conversion_requires_copy: bool,
    pub samples: u64,
    pub failures: u64,
}

impl SglangGfxGraphContract {
    pub fn new(enabled: bool, validation_enabled: bool, buckets: Vec<usize>) -> Self {
        Self {
            enabled,
            validation_enabled,
            buckets,
            decode_impl_path: "gfxgraph.sglang.decode".to_string(),
            prefill_impl_path: "gfxgraph.sglang.prefill".to_string(),
        }
    }

    pub fn from_env() -> Self {
        let disabled = std::env::var("SGLANG_DISABLE_GFXGRAPH")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let enabled = !disabled
            && std::env::var("SGLANG_RDNA2_KERNELS")
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false);
        let validation_enabled = std::env::var("SGLANG_GFXGRAPH_VALIDATE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        Self::new(enabled, validation_enabled, vec![1, 4, 8, 16, 32, 64, 128])
    }

    pub fn decode_node_spec(&self) -> GfxGraphNodeSpec {
        GfxGraphNodeSpec::new("sglang_decode", self.decode_impl_path.clone())
    }

    pub fn prefill_node_spec(&self) -> GfxGraphNodeSpec {
        GfxGraphNodeSpec::new("sglang_prefill", self.prefill_impl_path.clone())
    }

    pub fn router(&self) -> BucketRouterCore {
        BucketRouterCore::new(self.buckets.clone())
    }

    pub fn validator_config(&self) -> ValidatorConfig {
        ValidatorConfig::new(self.validation_enabled)
    }

    pub fn runner_state(&self) -> GraphRunnerState {
        GraphRunnerState::new(
            vec!["decode".to_string(), "prefill".to_string()],
            Vec::new(),
        )
    }

    pub fn capability_registry(&self) -> GraphCapabilityRegistry {
        GraphCapabilityRegistry::baseline()
    }

    pub fn layout_contract_for_tokens(&self, tokens: usize) -> Option<ShapeLayoutConversionPlan> {
        let shape = Shape::new(vec![tokens.max(1), 128]).ok()?;
        let layout = TensorLayout::row_major(shape.clone()).ok()?;
        ShapeLayoutConversionPlan::plan(
            &layout,
            shape,
            LayoutKind::RowMajor,
            Some(DTypeConversionContract::validate_only(
                DTypeKind::F16,
                DTypeKind::F16,
            )),
        )
        .ok()
    }

    pub fn status_for_shape(
        &self,
        shape_size: usize,
        stats: GfxGraphStatsSample,
    ) -> SglangGfxGraphStatus {
        let route = self.router().route(shape_size).ok();
        let registry = self.capability_registry();
        let conversion_requires_copy = self
            .layout_contract_for_tokens(shape_size)
            .map(|plan| plan.requires_copy)
            .unwrap_or(true);
        SglangGfxGraphStatus {
            enabled: self.enabled,
            validation_enabled: self.validator_config().should_validate(),
            buckets: self.buckets.clone(),
            decode_node: self.decode_node_spec().impl_path().to_string(),
            prefill_node: self.prefill_node_spec().impl_path().to_string(),
            routed_bucket: route.map(|(bucket, _)| bucket),
            routed_bucket_state: route.map(|(_, state)| state),
            registry_capabilities: registry.capabilities.len(),
            conversion_requires_copy,
            samples: stats.samples,
            failures: stats.failures,
        }
    }

    pub fn emit_contract_status(&self) {
        let status = self.status_for_shape(1, GfxGraphStatsSample::default());
        metrics::counter!("sgl_model_gateway_gfxgraph_contract_status_total").increment(1);
        metrics::gauge!("sgl_model_gateway_gfxgraph_contract_enabled").set(if status.enabled {
            1.0
        } else {
            0.0
        });
        metrics::gauge!("sgl_model_gateway_gfxgraph_validation_enabled")
            .set(if status.validation_enabled { 1.0 } else { 0.0 });
        tracing::info!(
            target: "sgl_model_gateway::gfxgraph",
            enabled = status.enabled,
            validation_enabled = status.validation_enabled,
            buckets = ?status.buckets,
            decode_node = %status.decode_node,
            prefill_node = %status.prefill_node,
            registry_capabilities = status.registry_capabilities,
            conversion_requires_copy = status.conversion_requires_copy,
            "gfxGRAPH contract status"
        );
    }
}

impl Default for SglangGfxGraphContract {
    fn default() -> Self {
        Self::new(false, false, vec![1, 4, 8, 16, 32, 64, 128])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_node_specs() {
        let contract = SglangGfxGraphContract::new(true, true, vec![1, 4, 8]);
        assert_eq!(contract.decode_node_spec().name(), "sglang_decode");
        assert_eq!(
            contract.prefill_node_spec().impl_path(),
            "gfxgraph.sglang.prefill"
        );
    }

    #[test]
    fn routes_shape_to_bucket() {
        let contract = SglangGfxGraphContract::new(true, false, vec![1, 4, 8, 16]);
        let status = contract.status_for_shape(9, GfxGraphStatsSample::new(3, 0));
        assert_eq!(status.routed_bucket, Some(16));
        assert_eq!(status.routed_bucket_state, Some(BucketState::NeedsWarmup));
        assert!(status.registry_capabilities > 0);
        assert!(!status.conversion_requires_copy);
        assert_eq!(status.samples, 3);
    }

    #[test]
    fn validator_and_runner_state_are_reachable() {
        let contract = SglangGfxGraphContract::new(true, true, vec![1]);
        assert!(contract.validator_config().should_validate());
        let runner = contract.runner_state();
        assert!(runner.is_known_branch("decode"));
        assert!(!runner.is_failed("decode"));
        assert!(contract.capability_registry().has("convert.shape_layout_dtype"));
        assert!(!contract.layout_contract_for_tokens(8).unwrap().requires_copy);
    }
}
