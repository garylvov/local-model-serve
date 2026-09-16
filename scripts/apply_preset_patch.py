#!/usr/bin/env python3
"""Apply a validated key=value patch to one [section] of a llama.cpp preset INI, in place,
line-by-line (NOT via configparser.write, which would strip every comment -- these files carry
load-bearing `; match:` lines that bin/llm's resolve() reads, plus operator documentation).

Used by bin/llm when the gateway queues a `set_preset` command for this peer (see
gateway/llm_gateway.py `/machines/<host>/preset`). Only ever called with peer-local, already-
validated arguments -- never with a raw shell string from the network. Re-validates everything
here too (known keys, GPU indices that exist, sane types) so a compromised or buggy gateway can
still not write arbitrary content into the file the router execs against.

Usage: apply_preset_patch.py <preset.ini> <section> <patch.json>
patch.json: {"key": "value", ...}  (value "" deletes the key; missing keys are left alone)
"""
import json
import os
import re
import subprocess
import sys

# Keys the Models tab is allowed to write. Anything else is rejected outright: this file is
# passed straight to `llama-server --models-preset`, so a bad key could execute nothing (llama.cpp
# only accepts known flags) but a bad *value* could still be abused (e.g. an arg-injection-looking
# string), hence the strict per-key validators below.
ALLOWED = {"device", "ctx-size", "parallel", "no-mmproj", "threads", "hf-repo",
           "image-min-tokens", "image-max-tokens", "n-gpu-layers"}
INT_KEYS = {"ctx-size", "parallel", "threads", "image-min-tokens", "image-max-tokens", "n-gpu-layers"}
BOOL_KEYS = {"no-mmproj"}
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(:[A-Za-z0-9._-]+)?$")
SECTION_RE = re.compile(r"^\[(.+)\]\s*$")
KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*=")


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5).stdout
        return sum(1 for ln in out.splitlines() if ln.startswith("GPU"))
    except Exception:
        return 0


def validate(key: str, value: str, n_gpu: int) -> str:
    if key not in ALLOWED:
        raise ValueError(f"key not allowed: {key}")
    if key == "device":
        if value == "":
            return value
        idxs = []
        for tok in value.split(","):
            tok = tok.strip()
            if not tok.startswith("CUDA") or not tok[4:].isdigit():
                raise ValueError(f"bad device token: {tok!r}")
            i = int(tok[4:])
            if i < 0 or i >= n_gpu:
                raise ValueError(f"GPU {i} does not exist on this machine ({n_gpu} GPUs)")
            idxs.append(i)
        if len(set(idxs)) != len(idxs):
            raise ValueError("duplicate GPU index")
        return ",".join(f"CUDA{i}" for i in idxs)
    if key in INT_KEYS:
        if value == "":
            return value
        if not re.fullmatch(r"-?\d+", value):
            raise ValueError(f"{key} must be an integer, got {value!r}")
        n = int(value)
        if n <= 0 or n > 8_000_000:
            raise ValueError(f"{key} out of range: {n}")
        return str(n)
    if key in BOOL_KEYS:
        if value not in ("", "true", "false"):
            raise ValueError(f"{key} must be true/false")
        return value
    if key == "hf-repo":
        if value != "" and not HF_REPO_RE.fullmatch(value):
            raise ValueError(f"bad hf-repo: {value!r}")
        return value
    return value


def patch_lines(lines: list, section: str, validated: dict) -> list:
    out, in_section, seen, found_section = [], False, set(), False
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SECTION_RE.match(line)
        if m:
            if in_section:
                # leaving the target section: append any keys that were not present
                for k, v in validated.items():
                    if k not in seen and v != "":
                        out.append(f"{k} = {v}\n")
            in_section = m.group(1) == section
            found_section = found_section or in_section
            out.append(line)
            i += 1
            continue
        if in_section:
            km = KEY_RE.match(line)
            if km and km.group(1) in validated:
                seen.add(km.group(1))
                v = validated[km.group(1)]
                if v != "":
                    out.append(f"{km.group(1)} = {v}\n")
                # v == "" -> drop the line (delete the key)
                i += 1
                continue
        out.append(line)
        i += 1
    if in_section:  # section was the last one in the file
        for k, v in validated.items():
            if k not in seen and v != "":
                out.append(f"{k} = {v}\n")
    if not found_section:
        raise ValueError(f"no [{section}] section in the preset")
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: apply_preset_patch.py <preset.ini> <section> <patch.json>", file=sys.stderr)
        return 2
    path, section, patch_json = sys.argv[1], sys.argv[2], sys.argv[3]
    patch = json.loads(patch_json)
    if not isinstance(patch, dict):
        raise ValueError("patch must be a JSON object")
    n_gpu = gpu_count()
    validated = {k: validate(k, str(v), n_gpu) for k, v in patch.items()}

    with open(path) as fh:
        lines = fh.readlines()
    new_lines = patch_lines(lines, section, validated)

    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(new_lines)
    os.replace(tmp, path)
    print(json.dumps({"ok": True, "section": section, "applied": validated}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001 - reported to caller as JSON on stderr
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        raise SystemExit(1)
