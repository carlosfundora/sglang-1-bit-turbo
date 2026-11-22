use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeError {
    EmptyShape,
    ZeroDimension { axis: usize },
    AxisOutOfBounds { axis: usize, rank: usize },
    ElementCountOverflow,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::EmptyShape => write!(f, "shape must contain at least one dimension"),
            ShapeError::ZeroDimension { axis } => write!(f, "dimension at axis {axis} is zero"),
            ShapeError::AxisOutOfBounds { axis, rank } => {
                write!(f, "axis {axis} is out of bounds for rank {rank}")
            }
            ShapeError::ElementCountOverflow => write!(f, "shape element count overflowed usize"),
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axis {
    pub name: String,
    pub index: usize,
}

impl Axis {
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Dim(pub usize);

impl Dim {
    pub fn new(value: usize) -> Result<Self, ShapeError> {
        if value == 0 {
            return Err(ShapeError::ZeroDimension { axis: 0 });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Rank(pub usize);

impl Rank {
    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Self, ShapeError> {
        if dims.is_empty() {
            return Err(ShapeError::EmptyShape);
        }
        for (axis, dim) in dims.iter().enumerate() {
            if *dim == 0 {
                return Err(ShapeError::ZeroDimension { axis });
            }
        }
        Ok(Self { dims })
    }

    pub fn from_dims_unchecked(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn rank_value(&self) -> Rank {
        Rank(self.rank())
    }

    pub fn dim(&self, axis: usize) -> Result<usize, ShapeError> {
        self.dims
            .get(axis)
            .copied()
            .ok_or(ShapeError::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            })
    }

    pub fn element_count(&self) -> Result<usize, ShapeError> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or(ShapeError::ElementCountOverflow)
    }

    pub fn ceil_axis_to(&self, axis: usize, bucket: usize) -> Result<Self, ShapeError> {
        let current = self.dim(axis)?;
        let mut dims = self.dims.clone();
        dims[axis] = current.next_multiple_of(bucket.max(1));
        Shape::new(dims)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchShape {
    pub batch: Dim,
    pub item: Shape,
}

impl BatchShape {
    pub fn new(batch: usize, item: Shape) -> Result<Self, ShapeError> {
        Ok(Self {
            batch: Dim::new(batch)?,
            item,
        })
    }

    pub fn full_shape(&self) -> Result<Shape, ShapeError> {
        let mut dims = Vec::with_capacity(self.item.rank() + 1);
        dims.push(self.batch.get());
        dims.extend_from_slice(self.item.dims());
        Shape::new(dims)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeKey {
    pub operation: String,
    pub dtype: String,
    pub dims: Vec<usize>,
    pub layout: String,
    pub backend: String,
}

impl ShapeKey {
    pub fn new(
        operation: impl Into<String>,
        dtype: impl Into<String>,
        shape: &Shape,
        layout: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            operation: operation.into(),
            dtype: dtype.into(),
            dims: shape.dims().to_vec(),
            layout: layout.into(),
            backend: backend.into(),
        }
    }

    pub fn stable_id(&self) -> String {
        let dims = self
            .dims
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x");
        format!(
            "{}:{}:{}:{}:{}",
            self.operation, self.dtype, dims, self.layout, self.backend
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphShape {
    pub name: String,
    pub shape: Shape,
    pub dynamic_axes: Vec<Axis>,
}

impl GraphShape {
    pub fn new(name: impl Into<String>, shape: Shape) -> Self {
        Self {
            name: name.into(),
            shape,
            dynamic_axes: Vec::new(),
        }
    }

    pub fn with_dynamic_axis(mut self, name: impl Into<String>, index: usize) -> Self {
        self.dynamic_axes.push(Axis::new(name, index));
        self
    }

    pub fn validate_dynamic_axes(&self) -> Result<(), ShapeError> {
        for axis in &self.dynamic_axes {
            self.shape.dim(axis.index)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_key_is_stable() {
        let shape = Shape::new(vec![2, 128, 64]).unwrap();
        let key = ShapeKey::new("decode", "fp16", &shape, "row_major", "rdna2");
        assert_eq!(key.stable_id(), "decode:fp16:2x128x64:row_major:rdna2");
    }

    #[test]
    fn shape_can_bucket_axis() {
        let shape = Shape::new(vec![3, 127]).unwrap();
        assert_eq!(shape.ceil_axis_to(1, 64).unwrap().dims(), &[3, 128]);
    }

    #[test]
    fn batch_shape_prepends_batch_dimension() {
        let shape = BatchShape::new(4, Shape::new(vec![128, 64]).unwrap()).unwrap();
        assert_eq!(shape.full_shape().unwrap().dims(), &[4, 128, 64]);
    }
}
