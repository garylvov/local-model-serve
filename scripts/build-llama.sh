#!/usr/bin/env bash
# Build llama.cpp with CUDA into vendor/llama.cpp.
# Usage: scripts/build-llama.sh [--ref <git-ref>] [--arch <cuda-archs>] [--jobs N]
# Env overrides: LLAMA_REF (default master), CUDA_ARCHS (default: auto-detect via
# nvidia-smi compute_cap, fallback 86), JOBS (default nproc).
# On Oscar, load CUDA first:  module load cuda/12.9.0-cinr
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/llama.cpp"
REF="${LLAMA_REF:-master}"
ARCH="${CUDA_ARCHS:-}"
JOBS="${JOBS:-$(nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  if command -v nvidia-smi >/dev/null; then
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | sort -u | paste -sd';')"
  fi
  ARCH="${ARCH:-86}"
fi

if ! command -v nvcc >/dev/null; then
  if command -v module >/dev/null 2>&1; then module load cuda/12.9.0-cinr || true; fi
fi
command -v nvcc >/dev/null || { echo "nvcc not found; load a CUDA toolkit" >&2; exit 1; }

mkdir -p "$ROOT/vendor"
if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --tags origin
git -C "$SRC" checkout -q "$REF"
if git -C "$SRC" symbolic-ref -q HEAD >/dev/null; then git -C "$SRC" pull -q --ff-only; fi

cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$SRC/build" --config Release -j "$JOBS" \
  --target llama-server llama-bench llama-cli llama-gguf-split

COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "$COMMIT" > "$ROOT/vendor/llama.cpp.commit"
echo "built llama.cpp $COMMIT (arch=$ARCH) -> $SRC/build/bin"
