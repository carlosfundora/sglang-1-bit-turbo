#!/usr/bin/env bash
# =============================================================================
# SGLang 1-Bit Turbo — ROCm/HIP Install Script
# =============================================================================
# Builds and installs sglang + sgl-kernel for AMD GPUs (ROCm).
# Auto-detects your GPU architecture and configures everything.
#
# Usage:
#   ./scripts/install_rocm.sh              # Full install (sglang + sgl-kernel)
#   ./scripts/install_rocm.sh --kernel     # Rebuild sgl-kernel only
#   ./scripts/install_rocm.sh --check      # Verify existing install
#
# Supported GPUs:
#   RDNA2:  RX 6700 XT, 6800, 6900 XT, etc. (gfx1030/1031/1032)
#   RDNA3:  RX 7900 XTX, 7800 XT, etc.      (gfx1100/1101/1102)
#   CDNA2:  MI200 series                      (gfx90a)
#   CDNA3:  MI300 series                      (gfx942)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ─── Argument parsing ────────────────────────────────────────────────────────
MODE="full"
for arg in "$@"; do
    case "$arg" in
        --kernel)  MODE="kernel" ;;
        --check)   MODE="check" ;;
        --help|-h) echo "Usage: $0 [--kernel|--check|--help]"; exit 0 ;;
        *) warn "Unknown arg: $arg" ;;
    esac
done

# ─── Prerequisites ───────────────────────────────────────────────────────────
check_prereqs() {
    info "Checking prerequisites..."

    # ROCm
    if ! command -v rocminfo &>/dev/null; then
        fail "ROCm not found. Install ROCm first: https://rocm.docs.amd.com/"
    fi
    ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    ok "ROCm $ROCM_VERSION"

    # Python
    PYTHON="${PYTHON:-python3}"
    if ! command -v "$PYTHON" &>/dev/null; then
        fail "Python not found. Set PYTHON env var or install python3."
    fi
    PY_VER=$("$PYTHON" --version 2>&1)
    ok "$PY_VER"

    # pip/uv
    if command -v uv &>/dev/null; then
        PIP_CMD="uv pip"
        ok "Using uv for package management"
    else
        PIP_CMD="$PYTHON -m pip"
        ok "Using pip for package management"
    fi

    # PyTorch with ROCm
    if ! "$PYTHON" -c "import torch; assert torch.version.hip" 2>/dev/null; then
        fail "PyTorch with ROCm support not found. Install pytorch-rocm first."
    fi
    TORCH_VER=$("$PYTHON" -c "import torch; print(f'PyTorch {torch.__version__} (HIP {torch.version.hip})')")
    ok "$TORCH_VER"
}

# ─── GPU Detection ───────────────────────────────────────────────────────────
detect_gpu() {
    info "Detecting AMD GPU..."

    # Try PyTorch first (most reliable)
    RAW_ARCH=$("$PYTHON" -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(props.gcnArchName.split(':')[0])
else:
    print('none')
" 2>/dev/null || echo "none")

    # Fallback to rocminfo
    if [[ "$RAW_ARCH" == "none" ]]; then
        RAW_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "none")
    fi

    if [[ "$RAW_ARCH" == "none" ]]; then
        fail "No AMD GPU detected. Check ROCm installation."
    fi

    GPU_NAME=$("$PYTHON" -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print('Unknown')
" 2>/dev/null || echo "Unknown AMD GPU")

    ok "GPU: $GPU_NAME ($RAW_ARCH)"

    # Normalize architecture for build
    case "$RAW_ARCH" in
        gfx1030|gfx1031|gfx1032|gfx1033|gfx1034|gfx1035|gfx1036)
            BUILD_ARCH="gfx1030"
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
            info "RDNA2 detected — normalizing to gfx1030, HSA_OVERRIDE_GFX_VERSION=10.3.0"
            ;;
        gfx1100|gfx1101|gfx1102|gfx1103)
            BUILD_ARCH="$RAW_ARCH"
            info "RDNA3 detected — using native $RAW_ARCH"
            warn "RDNA3 support is experimental. RDNA2 (gfx1030) is the primary target."
            ;;
        gfx90a)
            BUILD_ARCH="gfx90a"
            info "CDNA2 (MI200) detected"
            ;;
        gfx940|gfx941|gfx942)
            BUILD_ARCH="gfx942"
            info "CDNA3 (MI300) detected"
            ;;
        gfx950)
            BUILD_ARCH="gfx950"
            info "CDNA4 (MI350) detected"
            ;;
        *)
            warn "Unrecognized arch: $RAW_ARCH — attempting build with gfx1030 fallback"
            BUILD_ARCH="gfx1030"
            export HSA_OVERRIDE_GFX_VERSION=10.3.0
            ;;
    esac

    export PYTORCH_ROCM_ARCH="$BUILD_ARCH"
    export AMDGPU_TARGET="$BUILD_ARCH"
}

# ─── Install sglang ─────────────────────────────────────────────────────────
install_sglang() {
    info "Installing SGLang 1-Bit Turbo (editable mode)..."
    cd "$REPO_ROOT"
    $PIP_CMD install -e "python/[all]" 2>&1 | tail -5
    ok "SGLang installed"
}

# ─── Build sgl-kernel ────────────────────────────────────────────────────────
build_kernel() {
    info "Building sgl-kernel for $BUILD_ARCH..."
    cd "$REPO_ROOT/sgl-kernel"

    # Clean previous build
    rm -rf build/

    info "Compiling HIP kernels (this takes 2-5 minutes)..."
    "$PYTHON" setup_rocm.py develop 2>&1 | tail -10

    if [[ $? -ne 0 ]]; then
        fail "sgl-kernel build failed. Check errors above."
    fi
    ok "sgl-kernel built for $BUILD_ARCH"
}

# ─── Verify install ──────────────────────────────────────────────────────────
verify_install() {
    info "Verifying installation..."

    "$PYTHON" -c "
import sys

# Check sglang
try:
    import sglang
    print(f'  ✅ sglang {sglang.__version__}')
except ImportError:
    print('  ❌ sglang not found')
    sys.exit(1)

# Check sgl_kernel
try:
    import sgl_kernel
    print(f'  ✅ sgl_kernel loaded')
except ImportError:
    print('  ❌ sgl_kernel not found')
    sys.exit(1)

# Check GGUF ops (the key feature of this fork)
ops = ['ggml_dequantize', 'ggml_mul_mat_vec_a8', 'ggml_mul_mat_a8',
       'ggml_moe_a8', 'ggml_moe_a8_vec', 'ggml_moe_get_block_size']
missing = [op for op in ops if not hasattr(sgl_kernel, op)]
if missing:
    print(f'  ❌ Missing GGUF ops: {missing}')
    sys.exit(1)
print(f'  ✅ All {len(ops)} GGUF GPU kernels available')

# Check speculative sampling
spec_ops = ['top_k_renorm_prob', 'top_p_renorm_prob']
for op in spec_ops:
    if hasattr(sgl_kernel, op):
        print(f'  ✅ {op}')
    else:
        print(f'  ⚠️  {op} not found (Triton/PyTorch fallback will be used)')

# Check GPU
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  ✅ GPU: {name} ({mem:.1f} GB VRAM)')
else:
    print('  ⚠️  No GPU detected (CPU-only mode)')
"

    if [[ $? -eq 0 ]]; then
        echo ""
        ok "Installation verified! Ready to run."
        echo ""
        echo "  Quick start:"
        echo "    python -m sglang.launch_server \\"
        echo "      --model-path <path-to-gguf-model> \\"
        echo "      --port 30000"
        echo ""
    else
        fail "Verification failed. See errors above."
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   SGLang 1-Bit Turbo — ROCm Install             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

case "$MODE" in
    full)
        check_prereqs
        detect_gpu
        install_sglang
        build_kernel
        verify_install
        ;;
    kernel)
        check_prereqs
        detect_gpu
        build_kernel
        verify_install
        ;;
    check)
        check_prereqs
        detect_gpu
        verify_install
        ;;
esac
