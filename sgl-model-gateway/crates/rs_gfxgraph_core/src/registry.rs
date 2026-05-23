use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphCapabilityKind {
    Geometry,
    Shape,
    Layout,
    Wave,
    Convert,
    Runner,
    Validator,
    Adapter,
    Signal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCapability {
    pub id: String,
    pub kind: GraphCapabilityKind,
    pub description: String,
    pub stable: bool,
    pub requires_runtime: bool,
}

impl GraphCapability {
    pub fn new(
        id: impl Into<String>,
        kind: GraphCapabilityKind,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            description: description.into(),
            stable: true,
            requires_runtime: false,
        }
    }

    pub fn runtime(mut self) -> Self {
        self.requires_runtime = true;
        self
    }

    pub fn experimental(mut self) -> Self {
        self.stable = false;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCapabilityRegistry {
    pub registry_id: String,
    pub capabilities: Vec<GraphCapability>,
}

impl GraphCapabilityRegistry {
    pub fn baseline() -> Self {
        Self {
            registry_id: "rs_gfxgraph_core:baseline".to_string(),
            capabilities: vec![
                GraphCapability::new(
                    "geometry.point_bounds_polygon",
                    GraphCapabilityKind::Geometry,
                    "Point2/Point3, Bounds2/Bounds3, Polygon2, and distance basics",
                ),
                GraphCapability::new(
                    "shape.rank_batch_graph",
                    GraphCapabilityKind::Shape,
                    "Dim, Rank, Shape, BatchShape, ShapeKey, and GraphShape contracts",
                ),
                GraphCapability::new(
                    "layout.stride_tiled_paged",
                    GraphCapabilityKind::Layout,
                    "Row/column/channel/tiled/page layout and contiguity contracts",
                ),
                GraphCapability::new(
                    "wave.launch_occupancy",
                    GraphCapabilityKind::Wave,
                    "Architecture, wavefront, workgroup, launch, and occupancy hints",
                ),
                GraphCapability::new(
                    "convert.shape_layout_dtype",
                    GraphCapabilityKind::Convert,
                    "Shape/layout conversion plans, page transforms, and dtype contracts",
                ),
                GraphCapability::new(
                    "runner.state",
                    GraphCapabilityKind::Runner,
                    "Graph runner state vocabulary for engine integrations",
                )
                .runtime(),
                GraphCapability::new(
                    "validator.config",
                    GraphCapabilityKind::Validator,
                    "Validation-only contracts for graph capture and replay readiness",
                ),
                GraphCapability::new(
                    "adapter.kind",
                    GraphCapabilityKind::Adapter,
                    "Adapter capability vocabulary for Python, Node, engine, and native callers",
                )
                .runtime(),
                GraphCapability::new(
                    "signal.frequency_window",
                    GraphCapabilityKind::Signal,
                    "Frequency ranges, scalar ranges, spectral shapes, and windows",
                ),
            ],
        }
    }

    pub fn has(&self, id: &str) -> bool {
        self.capabilities.iter().any(|capability| capability.id == id)
    }

    pub fn by_kind(&self, kind: GraphCapabilityKind) -> Vec<&GraphCapability> {
        self.capabilities
            .iter()
            .filter(|capability| capability.kind == kind)
            .collect()
    }
}

impl Default for GraphCapabilityRegistry {
    fn default() -> Self {
        Self::baseline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_registry_names_conversion_capability() {
        let registry = GraphCapabilityRegistry::baseline();
        assert!(registry.has("convert.shape_layout_dtype"));
        assert_eq!(
            registry.by_kind(GraphCapabilityKind::Wave)[0].id,
            "wave.launch_occupancy"
        );
    }
}
