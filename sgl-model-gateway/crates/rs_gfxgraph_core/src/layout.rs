use serde::{Deserialize, Serialize};

use crate::shape::{Shape, ShapeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    RowMajor,
    ColumnMajor,
    ChannelsLast,
    Tiled,
    Paged,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Contiguity {
    Contiguous,
    Strided,
    Tiled,
    Paged,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrideSpec {
    strides: Vec<usize>,
}

impl StrideSpec {
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }

    pub fn row_major(shape: &Shape) -> Result<Self, ShapeError> {
        let mut strides = vec![1usize; shape.rank()];
        for axis in (0..shape.rank().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape.dim(axis + 1)?)
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(Self { strides })
    }

    pub fn column_major(shape: &Shape) -> Result<Self, ShapeError> {
        let mut strides = vec![1usize; shape.rank()];
        for axis in 1..shape.rank() {
            strides[axis] = strides[axis - 1]
                .checked_mul(shape.dim(axis - 1)?)
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(Self { strides })
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileLayoutSpec {
    pub tile_shape: Vec<usize>,
    pub tile_order: Vec<usize>,
}

impl TileLayoutSpec {
    pub fn new(tile_shape: Vec<usize>) -> Self {
        let tile_order = (0..tile_shape.len()).collect();
        Self {
            tile_shape,
            tile_order,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageLayoutSpec {
    pub page_axis: usize,
    pub page_size: usize,
    pub page_stride: usize,
}

impl PageLayoutSpec {
    pub fn new(page_axis: usize, page_size: usize, page_stride: usize) -> Self {
        Self {
            page_axis,
            page_size,
            page_stride,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLayout {
    pub kind: LayoutKind,
    pub shape: Shape,
    pub strides: StrideSpec,
    pub tile: Option<TileLayoutSpec>,
    pub page: Option<PageLayoutSpec>,
}

impl TensorLayout {
    pub fn row_major(shape: Shape) -> Result<Self, ShapeError> {
        let strides = StrideSpec::row_major(&shape)?;
        Ok(Self {
            kind: LayoutKind::RowMajor,
            shape,
            strides,
            tile: None,
            page: None,
        })
    }

    pub fn column_major(shape: Shape) -> Result<Self, ShapeError> {
        let strides = StrideSpec::column_major(&shape)?;
        Ok(Self {
            kind: LayoutKind::ColumnMajor,
            shape,
            strides,
            tile: None,
            page: None,
        })
    }

    pub fn tiled(
        shape: Shape,
        strides: StrideSpec,
        tile: TileLayoutSpec,
    ) -> Result<Self, ShapeError> {
        if tile.tile_shape.len() != shape.rank() {
            return Err(ShapeError::AxisOutOfBounds {
                axis: tile.tile_shape.len(),
                rank: shape.rank(),
            });
        }
        Ok(Self {
            kind: LayoutKind::Tiled,
            shape,
            strides,
            tile: Some(tile),
            page: None,
        })
    }

    pub fn paged(
        shape: Shape,
        strides: StrideSpec,
        page: PageLayoutSpec,
    ) -> Result<Self, ShapeError> {
        shape.dim(page.page_axis)?;
        Ok(Self {
            kind: LayoutKind::Paged,
            shape,
            strides,
            tile: None,
            page: Some(page),
        })
    }

    pub fn is_row_contiguous(&self) -> bool {
        StrideSpec::row_major(&self.shape)
            .map(|expected| expected == self.strides)
            .unwrap_or(false)
    }

    pub fn contiguity(&self) -> Contiguity {
        match self.kind {
            LayoutKind::RowMajor if self.is_row_contiguous() => Contiguity::Contiguous,
            LayoutKind::ColumnMajor | LayoutKind::ChannelsLast => Contiguity::Strided,
            LayoutKind::Tiled => Contiguity::Tiled,
            LayoutKind::Paged => Contiguity::Paged,
            LayoutKind::RowMajor => Contiguity::Strided,
            LayoutKind::Custom => Contiguity::Unknown,
        }
    }

    pub fn linear_offset(&self, indices: &[usize]) -> Result<usize, ShapeError> {
        if indices.len() != self.shape.rank() {
            return Err(ShapeError::AxisOutOfBounds {
                axis: indices.len(),
                rank: self.shape.rank(),
            });
        }

        let mut offset = 0usize;
        for (axis, index) in indices.iter().copied().enumerate() {
            let dim = self.shape.dim(axis)?;
            if index >= dim {
                return Err(ShapeError::AxisOutOfBounds {
                    axis: index,
                    rank: dim,
                });
            }
            offset = offset
                .checked_add(index * self.strides.strides()[axis])
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(offset)
    }

    pub fn needs_copy_for_graph_capture(&self) -> bool {
        !self.is_row_contiguous() || matches!(self.kind, LayoutKind::Custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutConversionPlan {
    pub source: LayoutKind,
    pub target: LayoutKind,
    pub requires_copy: bool,
    pub reason: String,
}

impl LayoutConversionPlan {
    pub fn from_layout(layout: &TensorLayout, target: LayoutKind) -> Self {
        let requires_copy = layout.kind != target
            || (target == LayoutKind::RowMajor && !layout.is_row_contiguous());
        let reason = if requires_copy {
            format!("convert {:?} layout to {:?}", layout.kind, target)
        } else {
            "layout already satisfies target".to_string()
        };
        Self {
            source: layout.kind,
            target,
            requires_copy,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_offsets_work() {
        let layout = TensorLayout::row_major(Shape::new(vec![2, 3, 4]).unwrap()).unwrap();
        assert_eq!(layout.strides.strides(), &[12, 4, 1]);
        assert_eq!(layout.linear_offset(&[1, 2, 3]).unwrap(), 23);
    }

    #[test]
    fn conversion_plan_marks_column_major_copy() {
        let layout = TensorLayout::column_major(Shape::new(vec![2, 3]).unwrap()).unwrap();
        let plan = LayoutConversionPlan::from_layout(&layout, LayoutKind::RowMajor);
        assert!(plan.requires_copy);
    }

    #[test]
    fn tiled_layout_reports_tiled_contiguity() {
        let shape = Shape::new(vec![8, 8]).unwrap();
        let layout = TensorLayout::tiled(
            shape,
            StrideSpec::new(vec![8, 1]),
            TileLayoutSpec::new(vec![4, 4]),
        )
        .unwrap();
        assert_eq!(layout.contiguity(), Contiguity::Tiled);
    }
}
