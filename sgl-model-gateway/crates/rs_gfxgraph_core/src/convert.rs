use serde::{Deserialize, Serialize};

use crate::layout::{LayoutConversionPlan, LayoutKind, StrideSpec, TensorLayout};
use crate::shape::{Shape, ShapeError};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DTypeKind {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DTypeConversionContract {
    pub source: DTypeKind,
    pub target: DTypeKind,
    pub validation_only: bool,
    pub lossy: bool,
    pub requires_runtime_kernel: bool,
}

impl DTypeConversionContract {
    pub fn validate_only(source: DTypeKind, target: DTypeKind) -> Self {
        let lossy = source != target;
        Self {
            source,
            target,
            validation_only: true,
            lossy,
            requires_runtime_kernel: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageTransform {
    pub page_size: usize,
    pub source_pages: usize,
    pub target_pages: usize,
    pub tail_items: usize,
}

impl PageTransform {
    pub fn plan(item_count: usize, source_page_size: usize, target_page_size: usize) -> Self {
        let source_pages = item_count.div_ceil(source_page_size.max(1));
        let target_pages = item_count.div_ceil(target_page_size.max(1));
        Self {
            page_size: target_page_size.max(1),
            source_pages,
            target_pages,
            tail_items: item_count % target_page_size.max(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrideTransform {
    pub source: StrideSpec,
    pub target: StrideSpec,
    pub requires_copy: bool,
}

impl StrideTransform {
    pub fn new(source: StrideSpec, target: StrideSpec) -> Self {
        let requires_copy = source != target;
        Self {
            source,
            target,
            requires_copy,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeLayoutConversionPlan {
    pub source_shape: Shape,
    pub target_shape: Shape,
    pub source_layout: LayoutKind,
    pub target_layout: LayoutKind,
    pub layout_plan: LayoutConversionPlan,
    pub dtype_contract: Option<DTypeConversionContract>,
    pub stride_transform: Option<StrideTransform>,
    pub page_transform: Option<PageTransform>,
    pub requires_copy: bool,
    pub reason: String,
}

impl ShapeLayoutConversionPlan {
    pub fn plan(
        source: &TensorLayout,
        target_shape: Shape,
        target_layout: LayoutKind,
        dtype_contract: Option<DTypeConversionContract>,
    ) -> Result<Self, ShapeError> {
        let layout_plan = LayoutConversionPlan::from_layout(source, target_layout);
        let target_stride = match target_layout {
            LayoutKind::ColumnMajor => StrideSpec::column_major(&target_shape)?,
            _ => StrideSpec::row_major(&target_shape)?,
        };
        let stride_transform = Some(StrideTransform::new(source.strides.clone(), target_stride));
        let shape_changes = source.shape.dims() != target_shape.dims();
        let dtype_requires_copy = dtype_contract
            .as_ref()
            .map(|contract| contract.source != contract.target && !contract.validation_only)
            .unwrap_or(false);
        let requires_copy = layout_plan.requires_copy || shape_changes || dtype_requires_copy;
        let reason = if requires_copy {
            "shape, layout, stride, or dtype contract requires a materialized transform".to_string()
        } else {
            "source shape and layout already satisfy target contract".to_string()
        };
        Ok(Self {
            source_shape: source.shape.clone(),
            target_shape,
            source_layout: source.kind,
            target_layout,
            layout_plan,
            dtype_contract,
            stride_transform,
            page_transform: None,
            requires_copy,
            reason,
        })
    }

    pub fn with_page_transform(mut self, page_transform: PageTransform) -> Self {
        if page_transform.source_pages != page_transform.target_pages {
            self.requires_copy = true;
        }
        self.page_transform = Some(page_transform);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::TensorLayout;

    #[test]
    fn plans_no_copy_for_matching_row_major_shape() {
        let layout = TensorLayout::row_major(Shape::new(vec![2, 4]).unwrap()).unwrap();
        let plan = ShapeLayoutConversionPlan::plan(
            &layout,
            Shape::new(vec![2, 4]).unwrap(),
            LayoutKind::RowMajor,
            None,
        )
        .unwrap();
        assert!(!plan.requires_copy);
    }

    #[test]
    fn page_transform_counts_target_pages() {
        let transform = PageTransform::plan(129, 64, 32);
        assert_eq!(transform.source_pages, 3);
        assert_eq!(transform.target_pages, 5);
    }
}
