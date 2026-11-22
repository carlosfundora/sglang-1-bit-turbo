#!/bin/bash

echo "Detecting ROCm Environment..."

if command -v rocminfo &> /dev/null; then
    rocminfo | grep "ROCm Version"
else
    echo "rocminfo not found. ROCm may not be installed."
fi

if command -v hipcc &> /dev/null; then
    hipcc --version | grep "HIP version"
else
    echo "hipcc not found."
fi

python3 --version
cat /etc/os-release | grep PRETTY_NAME

echo "GPU Architecture target:"
/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null || echo "Could not enumerate agents."
