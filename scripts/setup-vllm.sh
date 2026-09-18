#!/usr/bin/env bash
# Create (or rebuild) the vLLM venv with a torch build that matches THIS machine's driver.
#
# uv can pick the CUDA wheel index from the installed driver (`--torch-backend=auto`), which is
# what we want here: plain `pip install vllm` grabbed a CUDA 13 torch that the driver (CUDA 12.9)
# refuses with "The NVIDIA driver on your system is too old". With auto detection uv resolves the
# matching cuXXX wheels instead of us pinning versions per machine.
#
#   scripts/setup-vllm.sh [venv-dir] [vllm-spec]      # default: vendor/vllm-venv, vllm (latest)
#   LLM_TORCH_BACKEND=cu126 scripts/setup-vllm.sh     # force a backend instead of auto
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
VENV="${1:-$ROOT/vendor/vllm-venv}"
SPEC="${2:-vllm}"
BACKEND="${LLM_TORCH_BACKEND:-auto}"

command -v uv >/dev/null || { echo "uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo none)"
cuda="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 || true)"
echo "driver $drv, $cuda, torch backend: $BACKEND"

uv venv -q -p "${LLM_PY:-3.12}" "$VENV"
# ninja: vLLM shells out to it for its Triton/inductor JIT, and only looks on PATH
uv pip install -q -p "$VENV/bin/python" --torch-backend="$BACKEND" "$SPEC" ninja

"$VENV/bin/python" - <<'PY'
import torch, importlib.metadata as md
print(f"installed: vllm {md.version('vllm')}, torch {torch.__version__}, cuda {torch.version.cuda}")
print("torch sees GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot use CUDA on this machine - wrong wheel for the driver?")
PY
echo "ok: $VENV"
