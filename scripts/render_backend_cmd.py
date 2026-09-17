#!/usr/bin/env python3
"""Render a launch command for a non-llama.cpp backend from catalog/backends.yaml + one preset
section. Prints JSON: {backend, port, model_id, cmd, cwd, env, health_endpoint, models_endpoint}
on success, or {"error": "..."} (exit 1) for an unsupported combination -- caught by bin/llm and
turned into `llm: <error>` rather than a silently-dropped flag (see catalog/backends.yaml footer).

Usage: render_backend_cmd.py <preset.ini> <section> <backends.yaml> <root_dir> <port>
"""
import configparser
import hashlib
import json
import re
import sys

import yaml


def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(1)


def ini_section(preset_path, section):
    cp = configparser.ConfigParser(strict=False, delimiters=("=",), comment_prefixes=(";", "#"), interpolation=None)
    cp.optionxform = str
    text = open(preset_path).read()
    first = re.search(r"^\[", text, re.M)
    cp.read_string(text[first.start():] if first else "")
    g = dict(cp.items("*")) if cp.has_section("*") else {}
    if not cp.has_section(section):
        fail(f"no section [{section}] in {preset_path}")
    g.update(cp.items(section))
    return g


def main():
    preset_path, section, backends_path, root, port = sys.argv[1:6]
    g = ini_section(preset_path, section)
    backend = g.get("backend", "llamacpp")
    catalog = (yaml.safe_load(open(backends_path)) or {}).get("backends", {})
    eng = catalog.get(backend)
    if eng is None:
        fail(f"unknown backend '{backend}' (not in {backends_path}); known: {', '.join(catalog)}")
    if backend == "llamacpp":
        fail("llamacpp is served by the router (llama-server --models-preset), not render_backend_cmd")

    supports = eng.get("supports", {})

    def unsupported(cap):
        v = str(supports.get(cap, "")).strip().lower()
        return v in ("", "unknown") or v.startswith("unsupported")

    repo_full = g.get("hf-repo", "")
    repo, _, quant = repo_full.partition(":")
    if not repo:
        fail(f"section [{section}] has no hf-repo (backend={backend} needs a model to serve)")

    # --- reject unsupported combinations up front, with the exact repo/quant/flag named ---
    quant_note = str(supports.get("quant_selection", "")).lower()
    looks_like_gguf_quant = bool(quant and re.search(r"(?i)q\d|iq\d|k_m|k_s|k_xl|bf16$|f16$|f32$", quant))
    if looks_like_gguf_quant and "gguf" in quant_note and "unsupported" in quant_note:
        fail(f"{backend} cannot load GGUF quant '{repo_full}': {supports.get('quant_selection')}")
    spec_model = g.get("spec-draft-model") or g.get("model-draft")
    if spec_model and unsupported("speculative"):
        fail(f"{backend} does not support speculative decoding (section [{section}] sets spec-draft-model={spec_model}); "
             f"drop spec-draft-* for this section, or choose an engine whose backends.yaml lists a 'speculative' mechanism")
    mmproj_set = g.get("mmproj") or (g.get("no-mmproj", "false") != "true" and "vision" in section)
    if mmproj_set and unsupported("mmproj"):
        fail(f"{backend} has no mmproj/multimodal support (section [{section}] wants one): {supports.get('mmproj', 'not documented')}")

    device = g.get("device", "")
    gpu_ids = [d[4:] for d in device.split(",") if d.startswith("CUDA") and d[4:].isdigit()]
    if not gpu_ids:
        fail(f"section [{section}] has no CUDA device= (backend={backend} needs explicit GPU placement)")
    ctx_size = g.get("ctx-size", "8192")
    parallel = g.get("parallel", "1")
    model_id = section
    port = int(port)

    root_path = root.rstrip("/")
    env = {}
    for k, v in (eng.get("env") or {}).items():
        env[k] = v.format(gpu_ids=",".join(gpu_ids))

    fmt = dict(
        port=port, model_path=repo_full, repo=repo, quant=quant, gpu_ids=",".join(gpu_ids),
        gpu_count=len(gpu_ids), ctx_size=ctx_size, parallel=parallel,
        models_dir=f"{root_path}/models", extra_args=g.get("extra-args", ""),
        mmproj_path=g.get("mmproj", ""), spec_model=spec_model or "", spec_json="{}",
        model_id=model_id,
        venv=f"{root_path}/{eng.get('venv', '')}",
        gpu_mem_util=g.get("gpu-mem-util", "0.85"),
        tp_flag=(f"--tensor-parallel-size {len(gpu_ids)}" if len(gpu_ids) > 1 else ""),
        spec_flag=(f"--speculative-config '{{\"model\": \"{spec_model}\", \"num_speculative_tokens\": 3}}'" if spec_model else ""),
        bin=f"{root_path}/{eng.get('build', {}).get('dir', 'vendor/ds4')}",
    )
    cmd_tpl = eng.get("cmd")
    if not cmd_tpl:
        fail(f"backend '{backend}' has no cmd template in {backends_path} (build it first / add one)")
    try:
        cmd = " ".join(cmd_tpl.split()).format(**fmt)
        cmd = re.sub(r"\s+", " ", cmd).strip()
    except KeyError as e:
        fail(f"cmd template for {backend} references unknown variable {e}")

    out = {
        "backend": backend, "port": port, "model_id": model_id, "cmd": cmd, "cwd": root_path,
        "env": env, "health_endpoint": eng.get("health_endpoint", "/health"),
        "models_endpoint": eng.get("models_endpoint", "/v1/models"),
        "adapter_port": port + 900,  # deterministic offset: engine on `port`, its gateway-facing adapter on port+900
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
