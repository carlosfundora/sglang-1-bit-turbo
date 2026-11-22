use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeometryError {
    NotEnoughVertices { got: usize },
    EmptyBounds,
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::NotEnoughVertices { got } => {
                write!(f, "polygon requires at least 3 vertices, got {got}")
            }
            GeometryError::EmptyBounds => write!(f, "cannot build bounds from an empty point set"),
        }
    }
}

impl std::error::Error for GeometryError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds2 {
    pub min: Point2,
    pub max: Point2,
}

impl Bounds2 {
    pub fn from_points(points: &[Point2]) -> Result<Self, GeometryError> {
        let first = points.first().copied().ok_or(GeometryError::EmptyBounds)?;
        let mut min = first;
        let mut max = first;
        for point in points.iter().copied().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        Ok(Self { min, max })
    }

    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    pub fn contains(&self, point: Point2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds3 {
    pub min: Point3,
    pub max: Point3,
}

impl Bounds3 {
    pub fn from_points(points: &[Point3]) -> Result<Self, GeometryError> {
        let first = points.first().copied().ok_or(GeometryError::EmptyBounds)?;
        let mut min = first;
        let mut max = first;
        for point in points.iter().copied().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }
        Ok(Self { min, max })
    }

    pub fn extent(&self) -> Point3 {
        Point3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    pub fn contains(&self, point: Point3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon2 {
    pub vertices: Vec<Point2>,
}

impl Polygon2 {
    pub fn new(vertices: Vec<Point2>) -> Result<Self, GeometryError> {
        if vertices.len() < 3 {
            return Err(GeometryError::NotEnoughVertices {
                got: vertices.len(),
            });
        }
        Ok(Self { vertices })
    }

    pub fn bounds(&self) -> Bounds2 {
        Bounds2::from_points(&self.vertices).expect("polygon has vertices")
    }

    pub fn contains(&self, point: Point2) -> bool {
        let mut inside = false;
        let mut previous = self.vertices[self.vertices.len() - 1];

        for current in &self.vertices {
            let crosses_y = (current.y > point.y) != (previous.y > point.y);
            if crosses_y {
                let x_intersection = (previous.x - current.x) * (point.y - current.y)
                    / (previous.y - current.y)
                    + current.x;
                if point.x < x_intersection {
                    inside = !inside;
                }
            }
            previous = *current;
        }

        inside
    }

    pub fn distance_to_boundary(&self, point: Point2) -> f64 {
        let mut min_distance = f64::INFINITY;
        for index in 0..self.vertices.len() {
            let start = self.vertices[index];
            let end = self.vertices[(index + 1) % self.vertices.len()];
            min_distance = min_distance.min(point_to_segment_distance(point, start, end));
        }
        min_distance
    }
}

pub fn point_to_segment_distance(point: Point2, start: Point2, end: Point2) -> f64 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_squared = dx * dx + dy * dy;
    if length_squared == 0.0 {
        return point.distance(start);
    }

    let t =
        (((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared).clamp(0.0, 1.0);
    let projection = Point2::new(start.x + t * dx, start.y + t * dy);
    point.distance(projection)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_contains_and_distance_work() {
        let polygon = Polygon2::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ])
        .unwrap();

        assert!(polygon.contains(Point2::new(5.0, 5.0)));
        assert!(!polygon.contains(Point2::new(12.0, 5.0)));
        assert!((polygon.distance_to_boundary(Point2::new(5.0, 5.0)) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn bounds3_reports_extent() {
        let bounds =
            Bounds3::from_points(&[Point3::new(-1.0, 2.0, 0.0), Point3::new(3.0, 8.0, 5.0)])
                .unwrap();
        assert_eq!(bounds.extent(), Point3::new(4.0, 6.0, 5.0));
    }
}
