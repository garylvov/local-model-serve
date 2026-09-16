#!/usr/bin/env bash
# Pre-warm the Linux page cache for a (sharded) GGUF by reading every shard in parallel.
# llama.cpp's loader reads tensors on a single thread, which on network filesystems (NFS/GPFS)
# is far below what the filesystem can serve with many readers in flight. Warming the cache with
# one reader per shard, then loading with the default mmap mode, lets llama.cpp page from RAM.
#   usage: scripts/prewarm.sh <any shard of the model> [parallelism]
set -euo pipefail
first="$1"; par="${2:-16}"
dir="$(dirname "$first")"; stem="$(basename "$first" | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//')"
shards=( "$dir/$stem"-*-of-*.gguf ); [[ -e "${shards[0]}" ]] || shards=( "$first" )
total=$(du -cbL "${shards[@]}" | tail -1 | cut -f1)
echo "prewarm: ${#shards[@]} file(s), $((total/1073741824)) GiB, parallelism $par"
t0=$(date +%s)
printf '%s\n' "${shards[@]}" | xargs -P "$par" -I{} dd if={} of=/dev/null bs=16M status=none
dt=$(( $(date +%s) - t0 )); dt=$(( dt > 0 ? dt : 1 ))
echo "prewarm: done in ${dt}s ($(( total / dt / 1048576 )) MiB/s)"
