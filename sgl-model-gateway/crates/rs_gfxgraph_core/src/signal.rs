use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignalError {
    InvalidRange { min: f32, max: f32 },
    InvalidFrequencyRange { min_hz: f32, max_hz: f32 },
    ZeroSampleRate,
    ZeroLength,
    InvalidFftSize { fft_size: usize },
    BinOutOfBounds { bin: usize, bins: usize },
}

impl std::fmt::Display for SignalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalError::InvalidRange { min, max } => {
                write!(f, "range min {min} must be less than max {max}")
            }
            SignalError::InvalidFrequencyRange { min_hz, max_hz } => {
                write!(
                    f,
                    "frequency min {min_hz}Hz must be less than max {max_hz}Hz"
                )
            }
            SignalError::ZeroSampleRate => write!(f, "sample rate must be greater than zero"),
            SignalError::ZeroLength => write!(f, "length must be greater than zero"),
            SignalError::InvalidFftSize { fft_size } => {
                write!(f, "FFT size {fft_size} must be at least 2")
            }
            SignalError::BinOutOfBounds { bin, bins } => {
                write!(f, "spectral bin {bin} is out of bounds for {bins} bins")
            }
        }
    }
}

impl std::error::Error for SignalError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScalarRange {
    pub min: f32,
    pub max: f32,
}

impl ScalarRange {
    pub fn new(min: f32, max: f32) -> Result<Self, SignalError> {
        if !min.is_finite() || !max.is_finite() || min >= max {
            return Err(SignalError::InvalidRange { min, max });
        }
        Ok(Self { min, max })
    }

    pub fn contains(&self, value: f32) -> bool {
        value >= self.min && value <= self.max
    }

    pub fn normalize(&self, value: f32) -> f32 {
        ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrequencyRangeHz {
    pub min_hz: f32,
    pub max_hz: f32,
}

impl FrequencyRangeHz {
    pub fn new(min_hz: f32, max_hz: f32) -> Result<Self, SignalError> {
        if !min_hz.is_finite() || !max_hz.is_finite() || min_hz < 0.0 || min_hz >= max_hz {
            return Err(SignalError::InvalidFrequencyRange { min_hz, max_hz });
        }
        Ok(Self { min_hz, max_hz })
    }

    pub fn nyquist(sample_rate_hz: u32) -> Result<Self, SignalError> {
        if sample_rate_hz == 0 {
            return Err(SignalError::ZeroSampleRate);
        }
        Self::new(0.0, sample_rate_hz as f32 / 2.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleWindow {
    pub sample_rate_hz: u32,
    pub start_sample: u64,
    pub sample_count: u64,
}

impl SampleWindow {
    pub fn new(
        sample_rate_hz: u32,
        start_sample: u64,
        sample_count: u64,
    ) -> Result<Self, SignalError> {
        if sample_rate_hz == 0 {
            return Err(SignalError::ZeroSampleRate);
        }
        if sample_count == 0 {
            return Err(SignalError::ZeroLength);
        }
        Ok(Self {
            sample_rate_hz,
            start_sample,
            sample_count,
        })
    }

    pub fn end_sample(&self) -> u64 {
        self.start_sample.saturating_add(self.sample_count)
    }

    pub fn duration_seconds(&self) -> f64 {
        self.sample_count as f64 / self.sample_rate_hz as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowKind {
    Rectangular,
    Hann,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSpec {
    pub kind: WindowKind,
    pub length: usize,
    pub periodic: bool,
}

impl WindowSpec {
    pub fn new(kind: WindowKind, length: usize) -> Result<Self, SignalError> {
        if length == 0 {
            return Err(SignalError::ZeroLength);
        }
        Ok(Self {
            kind,
            length,
            periodic: true,
        })
    }

    pub fn symmetric(mut self) -> Self {
        self.periodic = false;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpectralShape {
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub bins: usize,
}

impl SpectralShape {
    pub fn new(sample_rate_hz: u32, fft_size: usize) -> Result<Self, SignalError> {
        if sample_rate_hz == 0 {
            return Err(SignalError::ZeroSampleRate);
        }
        if fft_size < 2 {
            return Err(SignalError::InvalidFftSize { fft_size });
        }
        Ok(Self {
            sample_rate_hz,
            fft_size,
            bins: fft_size / 2 + 1,
        })
    }

    pub fn bin_hz_step(&self) -> f32 {
        self.sample_rate_hz as f32 / self.fft_size as f32
    }

    pub fn hz_for_bin(&self, bin: usize) -> Result<f32, SignalError> {
        if bin >= self.bins {
            return Err(SignalError::BinOutOfBounds {
                bin,
                bins: self.bins,
            });
        }
        Ok(bin as f32 * self.bin_hz_step())
    }

    pub fn bin_for_hz(&self, hz: f32) -> usize {
        let bin = (hz.max(0.0) / self.bin_hz_step()).round() as usize;
        bin.min(self.bins.saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_range_normalizes_values() {
        let range = ScalarRange::new(-1.0, 1.0).unwrap();
        assert_eq!(range.normalize(-2.0), 0.0);
        assert_eq!(range.normalize(0.0), 0.5);
        assert_eq!(range.normalize(3.0), 1.0);
    }

    #[test]
    fn sample_window_reports_duration() {
        let window = SampleWindow::new(16_000, 320, 1_600).unwrap();
        assert_eq!(window.end_sample(), 1_920);
        assert!((window.duration_seconds() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn spectral_shape_maps_bins_and_hz() {
        let shape = SpectralShape::new(48_000, 1024).unwrap();
        assert_eq!(shape.bins, 513);
        assert_eq!(shape.bin_for_hz(1_000.0), 21);
        assert!((shape.hz_for_bin(21).unwrap() - 984.375).abs() < 0.001);
    }
}
