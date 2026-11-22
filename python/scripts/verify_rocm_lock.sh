#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if rg -i 'cuda|nvidia|cu12|cu129' uv.lock pyproject.toml; then
  echo "ROCm lock verification failed: forbidden accelerator packages are present." >&2
  exit 1
fi

if [[ -x .venv/bin/python ]]; then
  if uv pip list --python .venv/bin/python | rg -i 'cuda|nvidia|cu12|cu129'; then
    echo "ROCm environment verification failed: forbidden packages are installed." >&2
    exit 1
  fi

  .venv/bin/python - <<'PY'
import sys

try:
    import torch
except ImportError:
    sys.exit("ROCm environment verification failed: torch is not installed.")

if torch.version.cuda is not None:
    sys.exit(f"ROCm environment verification failed: torch CUDA version is {torch.version.cuda!r}.")

if getattr(torch.version, "hip", None) is None:
    sys.exit("ROCm environment verification failed: torch HIP version is missing.")

print(f"torch {torch.__version__}; HIP {torch.version.hip}; CUDA {torch.version.cuda}")
PY
fi

echo "ROCm lock verification passed."
