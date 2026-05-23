pub mod adapter;
pub mod convert;
pub mod error;
pub mod geometry;
pub mod layout;
pub mod registry;
pub mod router;
pub mod runner;
pub mod schema;
pub mod shape;
pub mod signal;
pub mod stats;
pub mod validator;
pub mod wave;

pub use adapter::GfxGraphAdapterKind;
pub use convert::{
    DTypeConversionContract, DTypeKind, PageTransform, ShapeLayoutConversionPlan, StrideTransform,
};
pub use error::{report_error, GfxGraphError};
pub use geometry::{
    point_to_segment_distance, Bounds2, Bounds3, GeometryError, Point2, Point3, Polygon2,
};
pub use layout::{
    Contiguity, LayoutConversionPlan, LayoutKind, PageLayoutSpec, StrideSpec, TensorLayout,
    TileLayoutSpec,
};
pub use registry::{GraphCapability, GraphCapabilityKind, GraphCapabilityRegistry};
pub use router::{BucketRouterCore, BucketState};
pub use runner::GraphRunnerState;
pub use schema::GfxGraphNodeSpec;
pub use shape::{Axis, BatchShape, Dim, GraphShape, Rank, Shape, ShapeError, ShapeKey};
pub use signal::{
    FrequencyRangeHz, SampleWindow, ScalarRange, SignalError, SpectralShape, WindowKind, WindowSpec,
};
pub use stats::GfxGraphStatsSample;
pub use validator::ValidatorConfig;
pub use wave::{
    GfxArch, HardwareProfile, KernelLaunchShape, OccupancyHint, WaveError, WavefrontSpec,
    WorkgroupShape,
};
