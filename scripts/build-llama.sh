#!/usr/bin/env bash
# Build llama.cpp with CUDA into vendor/llama.cpp.
# Usage: scripts/build-llama.sh [<variant>] [--ref <git-ref>] [--arch <cuda-archs>] [--jobs N]
# Variants come from catalog/builds.yaml: `master` -> vendor/llama.cpp, others -> vendor/llama.cpp-<variant>.
# Env overrides: LLAMA_REF (default master), CUDA_ARCHS (default: auto-detect via
# nvidia-smi compute_cap, fallback 86), JOBS (default nproc).
# On Oscar, load CUDA first:  module load cuda/12.9.0-cinr
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${LLAMA_VARIANT:-master}"
if [[ $# -gt 0 && "$1" != -* ]]; then VARIANT="$1"; shift; fi
BUILDS="$ROOT/catalog/builds.yaml"
yamlget() { sed -n "/^  $VARIANT:/,/^  [a-z]/p" "$BUILDS" | sed -n "s/^    $1: *//p" | head -1; }
REMOTE="$(yamlget remote)"; REMOTE="${REMOTE:-https://github.com/ggml-org/llama.cpp}"
REF_DEFAULT="$(yamlget ref)"; REF_DEFAULT="${REF_DEFAULT%%#*}"; REF_DEFAULT="${REF_DEFAULT%% *}"
SRC="$ROOT/$(yamlget dir)"; [[ "$SRC" == "$ROOT/" ]] && SRC="$ROOT/vendor/llama.cpp-$VARIANT"
REF="${LLAMA_REF:-${REF_DEFAULT:-master}}"
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
  git clone "$REMOTE" "$SRC"
fi
git -C "$SRC" fetch --tags origin "+refs/heads/*:refs/remotes/origin/*" "$REF" 2>/dev/null || git -C "$SRC" fetch --tags origin
git -C "$SRC" checkout -q "$REF" 2>/dev/null || git -C "$SRC" checkout -q FETCH_HEAD
if git -C "$SRC" symbolic-ref -q HEAD >/dev/null; then git -C "$SRC" pull -q --ff-only; fi

cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$SRC/build" --config Release -j "$JOBS" \
  --target llama-server llama-bench llama-cli llama-gguf-split llama-app

COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "$COMMIT" > "$SRC.commit"
echo "built llama.cpp [$VARIANT] $COMMIT (arch=$ARCH, $(du -sh "$SRC/build" | cut -f1) of build output) -> $SRC/build/bin"
