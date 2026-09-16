"""Find a preset model's weight files in the llama.cpp cache, and warm or evict them in the page cache.

llama.cpp reads weights on one thread, which is slow on network filesystems, and never tells the
kernel to drop a model's pages after unloading. Both matter once models are hundreds of GiB:
  * warm:  read every file in parallel so the loader finds the pages already in RAM
  * evict: POSIX_FADV_DONTNEED on every file, so an unloaded model's pages stop crowding out the
           next model's (no root needed; the kernel drops pages that nothing else maps)

    python3 scripts/weights_cache.py files <hf-repo[:quant]>            # print paths
    python3 scripts/weights_cache.py warm  <hf-repo[:quant]> [--min-gib N] [--jobs N]
    python3 scripts/weights_cache.py evict <hf-repo[:quant]>
"""
import concurrent.futures as cf
import glob
import os
import sys
import time

CACHE = os.environ.get("LLAMA_CACHE", "models")


def files(spec: str) -> list[str]:
    repo, _, quant = spec.partition(":")
    snap = os.path.join(CACHE, "models--" + repo.replace("/", "--"), "snapshots")
    paths = sorted(glob.glob(os.path.join(snap, "*", "**", "*.gguf"), recursive=True))
    paths = [p for p in paths if "mmproj" not in os.path.basename(p).lower()]
    if quant:
        q = quant.lower().removeprefix("ud-")
        paths = [p for p in paths if q in p.lower()]
    return [os.path.realpath(p) for p in paths]


def read_all(path: str) -> int:
    n, buf = 0, bytearray(16 << 20)
    with open(path, "rb", buffering=0) as f:
        while (k := f.readinto(buf)):
            n += k
    return n


def warm(spec: str, min_gib: float, jobs: int) -> None:
    ps = files(spec)
    total = sum(os.path.getsize(p) for p in ps)
    if not ps or total < min_gib * 2**30:
        return
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max(1, min(jobs, len(ps)))) as ex:
        list(ex.map(read_all, ps))
    dt = max(time.monotonic() - t0, 1e-3)
    print(f"warmed {len(ps)} file(s), {total / 2**30:.0f} GiB in {dt:.0f}s ({total / dt / 2**20:.0f} MiB/s)")


def evict(spec: str) -> None:
    ps = files(spec)
    for p in ps:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    if ps:
        print(f"evicted page cache for {len(ps)} file(s), {sum(os.path.getsize(p) for p in ps) / 2**30:.0f} GiB")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("files", "warm", "evict"):
        sys.exit(__doc__)
    cmd, spec, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    opt = lambda k, d: type(d)(rest[rest.index(k) + 1]) if k in rest else d
    if cmd == "files":
        print("\n".join(files(spec)))
    elif cmd == "warm":
        warm(spec, opt("--min-gib", 32.0), opt("--jobs", 16))
    else:
        evict(spec)
