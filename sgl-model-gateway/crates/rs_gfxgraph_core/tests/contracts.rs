use rs_gfxgraph_core::schema::GfxGraphNodeSpec;
use rs_gfxgraph_core::error::{GfxGraphError, report_error};

#[test]
fn node_spec_round_trips_through_json() {
    let spec = GfxGraphNodeSpec::new("decode", "gfxgraph.aiter_triton.flash_attention_decode");
    let json = serde_json::to_string(&spec).expect("serialize node spec");
    let parsed: GfxGraphNodeSpec = serde_json::from_str(&json).expect("parse node spec");
    assert_eq!(parsed.name(), "decode");
    assert_eq!(parsed.impl_path(), "gfxgraph.aiter_triton.flash_attention_decode");
}

#[test]
fn error_reporting_runs_without_panic() {
    let err = GfxGraphError::InvalidNode {
        name: "test_node".to_string(),
        reason: "invalid module path".to_string(),
    };
    report_error(&err, "TEST_CATEGORY");
}

