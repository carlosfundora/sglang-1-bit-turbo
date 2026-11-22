use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GfxArch {
    Gfx1030,
    Gfx1031,
    Gfx1100,
    Cuda,
    Cpu,
    Unknown,
}

impl GfxArch {
    pub fn from_arch_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("gfx1030") {
            Self::Gfx1030
        } else if lower.contains("gfx1031") {
            Self::Gfx1031
        } else if lower.contains("gfx1100") {
            Self::Gfx1100
        } else if lower.contains("cuda") {
            Self::Cuda
        } else if lower.contains("cpu") {
            Self::Cpu
        } else {
            Self::Unknown
        }
    }

    pub fn preferred_wave_size(self) -> Option<u32> {
        match self {
            GfxArch::Gfx1030 | GfxArch::Gfx1031 => Some(32),
            GfxArch::Gfx1100 => Some(32),
            GfxArch::Cuda => Some(32),
            GfxArch::Cpu | GfxArch::Unknown => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WavefrontSpec {
    pub arch: GfxArch,
    pub wave_size: u32,
    pub waves_per_eu: u32,
    pub matrix_instr_supported: bool,
}

impl WavefrontSpec {
    pub fn rdna2_gfx1030() -> Self {
        Self {
            arch: GfxArch::Gfx1030,
            wave_size: 32,
            waves_per_eu: 1,
            matrix_instr_supported: false,
        }
    }

    pub fn validate(&self) -> Result<(), WaveError> {
        if self.wave_size == 0 {
            return Err(WaveError::InvalidWaveSize(self.wave_size));
        }
        if self.waves_per_eu == 0 {
            return Err(WaveError::InvalidWavesPerEu(self.waves_per_eu));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkgroupShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupShape {
    pub fn new(x: u32, y: u32, z: u32) -> Result<Self, WaveError> {
        let shape = Self { x, y, z };
        shape.validate()?;
        Ok(shape)
    }

    pub fn threads(&self) -> u32 {
        self.x.saturating_mul(self.y).saturating_mul(self.z)
    }

    pub fn validate(&self) -> Result<(), WaveError> {
        if self.x == 0 || self.y == 0 || self.z == 0 {
            return Err(WaveError::InvalidWorkgroupShape(*self));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelLaunchShape {
    pub grid: WorkgroupShape,
    pub block: WorkgroupShape,
}

impl KernelLaunchShape {
    pub fn validate(&self, wave: WavefrontSpec) -> Result<(), WaveError> {
        wave.validate()?;
        self.grid.validate()?;
        self.block.validate()?;
        if self.block.threads() % wave.wave_size != 0 {
            return Err(WaveError::BlockNotWaveAligned {
                block_threads: self.block.threads(),
                wave_size: wave.wave_size,
            });
        }
        Ok(())
    }

    pub fn occupancy_hint(&self, wave: WavefrontSpec) -> Result<OccupancyHint, WaveError> {
        self.validate(wave)?;
        let waves_per_block = self.block.threads() / wave.wave_size;
        let blocks_per_cu = if waves_per_block == 0 {
            0
        } else {
            wave.waves_per_eu / waves_per_block
        }
        .max(1);
        Ok(OccupancyHint {
            waves_per_block,
            blocks_per_cu,
            occupancy_ratio: (waves_per_block as f32 / wave.waves_per_eu.max(1) as f32).min(1.0),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OccupancyHint {
    pub waves_per_block: u32,
    pub blocks_per_cu: u32,
    pub occupancy_ratio: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub arch: GfxArch,
    pub wavefront: WavefrontSpec,
    pub recommended_decode_block: WorkgroupShape,
    pub recommended_prefill_block: WorkgroupShape,
}

impl HardwareProfile {
    pub fn rdna2_gfx1030() -> Self {
        Self {
            name: "rdna2-gfx1030".to_string(),
            arch: GfxArch::Gfx1030,
            wavefront: WavefrontSpec::rdna2_gfx1030(),
            recommended_decode_block: WorkgroupShape { x: 32, y: 1, z: 1 },
            recommended_prefill_block: WorkgroupShape { x: 64, y: 2, z: 1 },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveError {
    InvalidWaveSize(u32),
    InvalidWavesPerEu(u32),
    InvalidWorkgroupShape(WorkgroupShape),
    BlockNotWaveAligned { block_threads: u32, wave_size: u32 },
}

impl std::fmt::Display for WaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaveError::InvalidWaveSize(value) => write!(f, "invalid wave size {value}"),
            WaveError::InvalidWavesPerEu(value) => write!(f, "invalid waves per eu {value}"),
            WaveError::InvalidWorkgroupShape(shape) => {
                write!(f, "invalid workgroup shape {:?}", shape)
            }
            WaveError::BlockNotWaveAligned {
                block_threads,
                wave_size,
            } => write!(
                f,
                "block thread count {block_threads} is not aligned to wave size {wave_size}"
            ),
        }
    }
}

impl std::error::Error for WaveError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rdna2_arch() {
        assert_eq!(GfxArch::from_arch_name("AMD gfx1030"), GfxArch::Gfx1030);
        assert_eq!(GfxArch::Gfx1030.preferred_wave_size(), Some(32));
    }

    #[test]
    fn launch_shape_requires_wave_alignment() {
        let launch = KernelLaunchShape {
            grid: WorkgroupShape::new(1, 1, 1).unwrap(),
            block: WorkgroupShape::new(16, 1, 1).unwrap(),
        };
        assert!(launch.validate(WavefrontSpec::rdna2_gfx1030()).is_err());
    }

    #[test]
    fn occupancy_hint_counts_waves_per_block() {
        let launch = KernelLaunchShape {
            grid: WorkgroupShape::new(1, 1, 1).unwrap(),
            block: WorkgroupShape::new(32, 1, 1).unwrap(),
        };
        let hint = launch
            .occupancy_hint(WavefrontSpec {
                waves_per_eu: 4,
                ..WavefrontSpec::rdna2_gfx1030()
            })
            .unwrap();
        assert_eq!(hint.waves_per_block, 1);
        assert_eq!(hint.blocks_per_cu, 4);
    }
}
