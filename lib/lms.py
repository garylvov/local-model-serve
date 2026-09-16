#!/usr/bin/env python3
"""lms: local-model-serve node launcher, registry writer, gateway and tunnel manager.

See README.md. No hostnames or GPU counts are hardcoded here; they come from
profiles/*.yaml, the environment, and nvidia-smi.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("lms: PyYAML is required (run scripts/bootstrap.sh, or: python3 -m pip install --user pyyaml)")

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "profiles"
MODELS_DIR = Path(os.environ.get("LMS_MODELS_DIR", ROOT / "models")).expanduser()
REGISTRY_DIR = Path(os.environ.get("LMS_REGISTRY_DIR", ROOT / "registry")).expanduser()
CONFIG_DIR = Path(os.environ.get("LMS_CONFIG_DIR", Path.home() / ".config" / "local-model-serve")).expanduser()
LLAMA_BIN = Path(os.environ.get("LMS_LLAMA_BIN", ROOT / "vendor" / "llama.cpp" / "build" / "bin")).expanduser()
HOST = os.environ.get("LMS_HOST", socket.gethostname().split(".")[0])
# Address other machines use to reach this node (defaults to short hostname).
ADVERTISE = os.environ.get("LMS_ADVERTISE_HOST", HOST)
RUN_DIR = ROOT / "run" / HOST
GATEWAY_PORT = int(os.environ.get("LMS_GATEWAY_PORT", "4000"))
STALE_SECONDS = int(os.environ.get("LMS_REGISTRY_STALE_SECONDS", "600"))
RESERVED = {"build", "gateway", "tunnel"}

BACKEND_KEY = CONFIG_DIR / "backend-key"
MASTER_KEY = CONFIG_DIR / "gateway-master-key"
AUTH_ENV = CONFIG_DIR / "auth.env"
CF_ACCESS_ENV = CONFIG_DIR / "cf-access.env"
TUNNEL_TOKEN = CONFIG_DIR / "tunnel-token"


def log(msg: str) -> None:
    print(f"[lms {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"lms: {msg}", file=sys.stderr)
    sys.exit(code)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- secrets

def private_write(path: Path, content: str) -> None:
    """Write a file readable only by the owner (dir 0700, file 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(path, 0o600)


def read_secret(path: Path) -> str:
    if not path.exists():
        die(f"missing {path}; run `lms auth init` (or copy it from the gateway host)")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        die(f"{path} has mode {oct(mode)}; expected 0600 (chmod 600 {path})")
    return path.read_text().strip()


def read_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    read_secret(path)  # mode check
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out


def ensure_key(path: Path, prefix: str) -> bool:
    if path.exists():
        read_secret(path)
        return False
    private_write(path, prefix + secrets.token_urlsafe(32) + "\n")
    return True


# --------------------------------------------------------------------------- hardware

def detect_gpus(fake: str | None = None) -> list[dict]:
    """Return visible GPUs: [{index, total_mib, free_mib, bus, numa}].

    `fake` = "N" or "NxMIB" fakes an inventory (for dry-runs). CUDA_VISIBLE_DEVICES
    (numeric form) filters and orders the physical GPUs like CUDA does.
    """
    if fake:
        n, _, mib = fake.partition("x")
        mib = int(mib or 24000)
        return [{"index": i, "total_mib": mib, "free_mib": mib - 500, "bus": None, "numa": None} for i in range(int(n))]
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,pci.bus_id", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, tot, free, bus = [x.strip() for x in line.split(",")]
        numa = None
        sysfs = Path("/sys/bus/pci/devices") / bus.lower()[-12:]
        try:
            numa = int((sysfs / "numa_node").read_text())
            numa = None if numa < 0 else numa
        except (OSError, ValueError):
            pass
        gpus.append({"index": int(idx), "total_mib": int(tot), "free_mib": int(free), "bus": bus, "numa": numa})
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and all(p.strip().isdigit() for p in cvd.split(",") if p.strip()):
        by_idx = {g["index"]: g for g in gpus}
        gpus = [by_idx[int(p)] for p in cvd.split(",") if p.strip() and int(p) in by_idx]
    return gpus


def ram_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0


def numa_nodes() -> int:
    return len(list(Path("/sys/devices/system/node").glob("node[0-9]*")))


# --------------------------------------------------------------------------- profiles

def profile_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    p = PROFILES_DIR / f"{name}.yaml"
    if not p.exists():
        die(f"unknown profile {name!r}; available: {', '.join(list_profiles())}")
    return p


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str) -> dict:
    path = profile_path(name)
    prof = yaml.safe_load(path.read_text()) or {}
    prof["name"] = path.stem
    defaults = prof.get("defaults", {}) or {}
    insts = {}
    for iname, spec in (prof.get("instances") or {}).items():
        if iname in RESERVED or iname.startswith("dl-") or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", iname):
            die(f"profile {path.name}: invalid instance name {iname!r}")
        merged = {**defaults, **(spec or {})}
        merged["args"] = list(defaults.get("args", []) or []) + list((spec or {}).get("args", []) or [])
        merged["name"] = iname
        for req in ("alias", "model", "gpus", "port"):
            if req not in merged:
                die(f"profile {path.name}: instance {iname} missing {req!r}")
        insts[iname] = merged
    prof["instances"] = insts
    return prof


def model_file(inst: dict) -> Path:
    m = inst["model"]
    p = Path(os.path.expandvars(m["file"])).expanduser()
    if p.is_absolute():
        return p
    return MODELS_DIR / m["repo"] / p


def pick_auto(gpus: list[dict], ram: float, verbose: bool = True) -> str:
    """Choose the highest-priority profile whose `auto` requirements hold."""
    n = len(gpus)
    min_free = min((g["free_mib"] for g in gpus), default=0)
    cands = []
    for name in list_profiles():
        prof = yaml.safe_load((PROFILES_DIR / f"{name}.yaml").read_text()) or {}
        a = prof.get("auto")
        if not a:
            continue
        need = int(a.get("gpus", 1))
        ok_gpus = n >= need if a.get("gpus_at_least") else n == need
        reasons = []
        if not ok_gpus:
            reasons.append(f"needs {'>=' if a.get('gpus_at_least') else '=='}{need} GPUs")
        if min_free < int(a.get("min_free_vram_mib", 0)):
            reasons.append(f"needs {a['min_free_vram_mib']} MiB free/GPU")
        if int(a.get("max_free_vram_mib", 10**9)) < min_free:
            reasons.append(f"GPUs larger than {a['max_free_vram_mib']} MiB (a bigger profile fits)")
        if ram < float(a.get("min_ram_gib", 0)):
            reasons.append(f"needs {a['min_ram_gib']} GiB RAM")
        prio = int(a.get("priority", 0))
        if verbose:
            print(f"  {name:<20} prio={prio:<4} {'OK' if not reasons else 'no: ' + '; '.join(reasons)}")
        if not reasons:
            cands.append((prio, need, name))
    if not cands:
        die("auto: no profile matches this machine; write profiles/<name>.yaml (see README)")
    cands.sort(reverse=True)
    return cands[0][2]


def numa_prefix(inst: dict, gpus: list[dict]) -> list[str]:
    numa = inst.get("numa", "auto")
    if numa in (None, "none", False) or not shutil.which("numactl") or numa_nodes() < 2:
        return []
    if numa == "auto":
        nodes = set()
        for i in inst["gpus"]:
            if i < len(gpus):
                nodes.add(gpus[i]["numa"])
        numa = nodes.pop() if len(nodes) == 1 and None not in nodes else "interleave"
    if numa == "interleave":
        return ["numactl", "--interleave=all"]
    return ["numactl", f"--cpunodebind={int(numa)}", f"--membind={int(numa)}"]


def server_cmd(inst: dict, gpus: list[dict]) -> tuple[list[str], dict]:
    parallel = int(inst.get("parallel", 1))
    ctx = int(inst.get("ctx_per_slot", 32768)) * parallel
    kv = inst.get("kv_type", "f16")
    cmd = numa_prefix(inst, gpus) + [
        str(LLAMA_BIN / "llama-server"),
        "-m", str(model_file(inst)),
        "--alias", inst["alias"],
        "--host", inst.get("bind", "0.0.0.0"),
        "--port", str(inst["port"]),
        "--api-key-file", str(BACKEND_KEY),
        "-ngl", str(inst.get("ngl", 99)),
        "-fa", "on",
        "--jinja",
        "--metrics",
        "-c", str(ctx),
        "-np", str(parallel),
        "-ctk", kv, "-ctv", kv,
    ]
    if inst.get("n_cpu_moe"):
        cmd += ["--n-cpu-moe", str(inst["n_cpu_moe"])]
    cmd += [str(a) for a in inst.get("args", [])]
    env = dict(os.environ)
    visible = [g["index"] for g in gpus] if gpus and gpus[0]["bus"] else list(range(max(inst["gpus"]) + 1))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(visible[i]) for i in inst["gpus"] if i < len(visible))
    return cmd, env


# --------------------------------------------------------------------------- tmux

def tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, check=check)


def session_exists(name: str) -> bool:
    return tmux("has-session", "-t", f"={name}").returncode == 0


def start_session(name: str, argv: list[str], cwd: Path = ROOT) -> None:
    tmux("new-session", "-d", "-s", name, "-c", str(cwd), shlex.join(argv), check=True)


def stop_session(name: str, grace: float = 60.0) -> None:
    if not session_exists(name):
        return
    tmux("send-keys", "-t", f"={name}", "C-c")
    deadline = time.time() + grace
    while time.time() < deadline and session_exists(name):
        time.sleep(1)
    if session_exists(name):
        tmux("kill-session", "-t", f"={name}")


# --------------------------------------------------------------------------- registry

def reg_path(instance: str) -> Path:
    return REGISTRY_DIR / f"{HOST}-{instance}.json"


def http_get(url: str, key: str | None = None, timeout: float = 5.0, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def read_registry(include_stale: bool = False) -> list[dict]:
    out = []
    for p in sorted(REGISTRY_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if d.get("kind") != "backend":
            continue
        age = time.time() - d.get("ts_epoch", 0)
        d["_stale"] = age > STALE_SECONDS
        d["_file"] = p.name
        if include_stale or not d["_stale"]:
            out.append(d)
    return out


def read_static() -> list[dict]:
    p = REGISTRY_DIR / "static.yaml"
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return data.get("backends", []) or []


# --------------------------------------------------------------------------- commands: node

def cmd_up(a) -> None:
    fake = a.fake_gpus
    gpus = detect_gpus(fake)
    name = a.profile
    if name == "auto":
        ram = a.fake_ram_gib if a.fake_ram_gib is not None else ram_gib()
        print(f"auto: {len(gpus)} GPU(s), min free VRAM {min((g['free_mib'] for g in gpus), default=0)} MiB, RAM {ram:.0f} GiB")
        name = pick_auto(gpus, ram)
        print(f"auto: selected profile {name!r}")
    prof = load_profile(name)
    insts = [prof["instances"][i] for i in (a.only or prof["instances"].keys())]
    for inst in insts:
        if max(inst["gpus"]) >= len(gpus) and not a.dry_run:
            die(f"{inst['name']}: profile wants GPU {max(inst['gpus'])} but only {len(gpus)} visible")
        cmd, env = server_cmd(inst, gpus)
        print(f"\n# instance {inst['name']} (alias {inst['alias']}) -> http://{ADVERTISE}:{inst['port']}")
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {shlex.join(cmd)}")
        if a.dry_run:
            continue
        sess = f"lms-{inst['name']}"
        if session_exists(sess):
            print(f"{sess}: already running; refusing to double-start (use `lms down {inst['name']}` first)")
            continue
        if not model_file(inst).exists():
            die(f"{inst['name']}: model file missing: {model_file(inst)} (run `lms fetch {name}`)")
        if not (LLAMA_BIN / "llama-server").exists():
            die(f"llama-server not built at {LLAMA_BIN} (run scripts/build-llama.sh)")
        if ensure_key(BACKEND_KEY, "lms-backend-"):
            log(f"generated backend key at {BACKEND_KEY}")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        start_session(sess, [str(ROOT / "bin" / "lms"), "_serve", str(profile_path(name)), inst["name"]])
        print(f"{sess}: started; logs: {RUN_DIR / (inst['name'] + '.log')}")
    if not a.dry_run:
        state = RUN_DIR / "profile"
        state.write_text(name + "\n")


def cmd_serve(a) -> None:
    """Internal: run inside tmux. Starts llama-server, registers on health, deregisters on exit."""
    prof = load_profile(a.profile)
    inst = prof["instances"][a.instance]
    gpus = detect_gpus()
    cmd, env = server_cmd(inst, gpus)
    logf = RUN_DIR / f"{inst['name']}.log"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(logf, "a") as lf:
        lf.write(f"\n==== {now_iso()} lms _serve {prof['name']}/{inst['name']}\n"
                 f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {shlex.join(cmd)}\n")
    lf = open(logf, "a")
    child = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    rp = reg_path(inst["name"])
    stopping = {"flag": False}

    def stop(signum, _frame):
        stopping["flag"] = True
        log(f"signal {signum}: stopping llama-server")
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)
    key = read_secret(BACKEND_KEY)
    healthy = False
    started = time.time()
    last_write = 0.0
    try:
        while child.poll() is None:
            ok = False
            try:
                st, _ = http_get(f"http://127.0.0.1:{inst['port']}/health", key, timeout=3)
                ok = st == 200
            except (urllib.error.URLError, OSError, ValueError):
                ok = False
            if ok and not stopping["flag"] and (not healthy or time.time() - last_write > 60):
                if not healthy:
                    log(f"healthy after {time.time() - started:.0f}s; registering {rp.name}")
                REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
                tmp = rp.with_suffix(".tmp")
                tmp.write_text(json.dumps({
                    "kind": "backend", "host": HOST, "instance": inst["name"],
                    "url": f"http://{ADVERTISE}:{inst['port']}", "alias": inst["alias"],
                    "profile": prof["name"], "parallel": int(inst.get("parallel", 1)),
                    "ctx_per_slot": int(inst.get("ctx_per_slot", 0)),
                    "ts": now_iso(), "ts_epoch": int(time.time()),
                }, indent=2) + "\n")
                tmp.replace(rp)
                healthy, last_write = True, time.time()
            elif not ok and healthy:
                log("health check failed; deregistering")
                rp.unlink(missing_ok=True)
                healthy = False
            time.sleep(5)
    finally:
        rp.unlink(missing_ok=True)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(child.pid, signal.SIGKILL)
        log(f"llama-server exited rc={child.returncode}; deregistered")


def node_sessions() -> list[str]:
    r = tmux("list-sessions", "-F", "#{session_name}")
    names = r.stdout.split() if r.returncode == 0 else []
    return [n for n in names if n.startswith("lms-") and n[4:] not in RESERVED and not n.startswith("lms-dl-")]


def cmd_down(a) -> None:
    targets = [f"lms-{i}" for i in a.instances] if a.instances else node_sessions()
    for s in targets:
        log(f"stopping {s}")
        stop_session(s)
        reg_path(s[4:]).unlink(missing_ok=True)
    if not a.instances:
        for p in REGISTRY_DIR.glob(f"{HOST}-*.json"):
            p.unlink(missing_ok=True)
        (RUN_DIR / "profile").unlink(missing_ok=True)


def cmd_status(a) -> None:
    prof = (RUN_DIR / "profile").read_text().strip() if (RUN_DIR / "profile").exists() else "-"
    print(f"host {HOST} (advertise {ADVERTISE}); profile: {prof}")
    key = BACKEND_KEY.read_text().strip() if BACKEND_KEY.exists() else None
    regs = {(d["host"], d["instance"]): d for d in read_registry(include_stale=True)}
    for s in node_sessions():
        d = regs.get((HOST, s[4:]))
        health = "-"
        if d:
            try:
                st, body = http_get(d["url"].replace(ADVERTISE, "127.0.0.1", 1) + "/health", key, 3)
                health = f"{st}"
            except Exception as e:  # noqa: BLE001
                health = f"ERR {type(e).__name__}"
        print(f"  {s:<22} registered={'yes' if d else 'no (starting/unhealthy)'} health={health}")
    print("registry (all nodes):")
    for d in regs.values():
        print(f"  {d['_file']:<32} {d['alias']:<18} {d['url']:<28} ts={d['ts']}{' STALE' if d['_stale'] else ''}")
    for b in read_static():
        print(f"  static:{b.get('name', '?'):<25} {b.get('alias', '?'):<18} {b.get('url', '?')}")
    g = "up" if session_exists("lms-gateway") else "down"
    print(f"gateway (this host): {g}; tunnel (this host): {'up' if session_exists('lms-tunnel') else 'down'}")


def cmd_logs(a) -> None:
    f = RUN_DIR / (f"{a.instance}.log")
    if not f.exists():
        die(f"no log at {f}")
    os.execvp("tail", ["tail", "-n", str(a.lines)] + (["-F"] if a.follow else []) + [str(f)])


def cmd_fetch(a) -> None:
    prof = load_profile(a.profile)
    hf = shutil.which("hf") or str(Path.home() / ".local/bin/hf")
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(MODELS_DIR / ".hf-home"))
    for inst in prof["instances"].values():
        if a.only and inst["name"] not in a.only:
            continue
        m = inst["model"]
        if model_file(inst).exists() and not a.force:
            print(f"{inst['name']}: present ({model_file(inst)})")
            continue
        argv = [hf, "download", m["repo"]]
        inc = m.get("include") or [m["file"]]
        for pat in ([inc] if isinstance(inc, str) else inc):
            argv += ["--include", pat]
        argv += ["--local-dir", str(MODELS_DIR / m["repo"])]
        print(shlex.join(argv))
        if a.dry_run:
            continue
        if a.tmux:
            sess = f"lms-dl-{inst['name']}"
            if session_exists(sess):
                print(f"{sess}: already running")
                continue
            logp = MODELS_DIR / "logs" / f"dl-{inst['name']}.log"
            logp.parent.mkdir(parents=True, exist_ok=True)
            tmux("new-session", "-d", "-s", sess, "-e", f"HF_HOME={env['HF_HOME']}",
                 f"{shlex.join(argv)} 2>&1 | tee -a {shlex.quote(str(logp))}", check=True)
            print(f"{sess}: started (log {logp})")
        else:
            subprocess.run(argv, env=env, check=True)


# --------------------------------------------------------------------------- auth

def cmd_auth(a) -> None:
    if a.auth_cmd == "init":
        for path, prefix in ((BACKEND_KEY, "lms-backend-"), (MASTER_KEY, "sk-lms-")):
            print(f"{path}: {'generated' if ensure_key(path, prefix) else 'exists'} (0600)")
        base = a.base_url.rstrip("/")
        if AUTH_ENV.exists() and not a.force:
            env = read_env_file(AUTH_ENV)
            if env.get("LMS_BASE_URL") != base:
                print(f"{AUTH_ENV}: exists with LMS_BASE_URL={env.get('LMS_BASE_URL')} (use --force to rewrite)")
            else:
                print(f"{AUTH_ENV}: exists")
            return
        private_write(AUTH_ENV, auth_env_text(base, read_secret(MASTER_KEY)))
        print(f"{AUTH_ENV}: written (0600); LMS_BASE_URL={base}")
    elif a.auth_cmd == "print-client":
        env = read_env_file(AUTH_ENV)
        if not env:
            die(f"missing {AUTH_ENV}; run `lms auth init` on the gateway host")
        text = auth_env_text(env["LMS_BASE_URL"], env["LMS_API_KEY"], a.host)
        if sys.stdout.isatty() and not a.stdout:
            die("refusing to print a key to a terminal (scrollback/logs). Pipe it instead, e.g.\n"
                f"  lms auth print-client --host {a.host or '<name>'} | ssh <name> "
                "'umask 077; mkdir -p ~/.config/local-model-serve && cat > ~/.config/local-model-serve/auth.env'\n"
                "or pass --stdout if you really want it on screen.")
        sys.stdout.write(text)


def auth_env_text(base: str, key: str, host: str | None = None) -> str:
    who = f" for {host}" if host else ""
    return (f"# local-model-serve client credentials{who}; generated {now_iso()}. Keep mode 0600.\n"
            f"LMS_BASE_URL={base}\n"
            f"LMS_OPENAI_BASE_URL={base}/v1\n"
            f"LMS_ANTHROPIC_BASE_URL={base}\n"
            f"LMS_API_KEY={key}\n")


# --------------------------------------------------------------------------- gateway

def gateway_config() -> tuple[dict, list[str]]:
    """Build a LiteLLM config from registry/*.json + registry/static.yaml. Secrets stay as os.environ/ refs."""
    model_list, notes = [], []
    for d in read_registry():
        model_list.append({
            "model_name": d["alias"],
            "litellm_params": {
                "model": f"openai/{d['alias']}",
                "api_base": d["url"] + "/v1",
                "api_key": "os.environ/LMS_BACKEND_KEY",
                "max_parallel_requests": d.get("parallel", 1),
            },
            "model_info": {"id": f"{d['host']}-{d['instance']}"},
        })
        notes.append(f"{d['alias']} <- {d['url']} ({d['_file']})")
    for b in read_static():
        params = {
            "model": f"openai/{b['alias']}",
            "api_base": b["url"].rstrip("/") + "/v1",
            "api_key": f"os.environ/{b.get('api_key_env', 'LMS_BACKEND_KEY')}",
        }
        if b.get("parallel"):
            params["max_parallel_requests"] = int(b["parallel"])
        if b.get("cf_access"):
            pre = b.get("cf_access_env_prefix", "")
            params["extra_headers"] = {
                "CF-Access-Client-Id": f"os.environ/{pre}CF_ACCESS_CLIENT_ID",
                "CF-Access-Client-Secret": f"os.environ/{pre}CF_ACCESS_CLIENT_SECRET",
            }
        model_list.append({"model_name": b["alias"], "litellm_params": params,
                           "model_info": {"id": f"static-{b.get('name', b['alias'])}"}})
        notes.append(f"{b['alias']} <- {b['url']} (static:{b.get('name', '?')})")
    cfg = {
        "model_list": model_list,
        "router_settings": {"routing_strategy": "least-busy", "num_retries": 1, "timeout": 1800,
                            "allowed_fails": 2, "cooldown_time": 30},
        "litellm_settings": {"drop_params": True, "request_timeout": 1800},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }
    return cfg, notes


def resolve_env_refs(obj):
    if isinstance(obj, dict):
        return {k: resolve_env_refs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env_refs(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("os.environ/"):
        v = os.environ.get(obj[len("os.environ/"):])
        if v is None:
            die(f"gateway: environment reference {obj} is unset (check {CF_ACCESS_ENV})")
        return v
    return obj


def cmd_gateway(a) -> None:
    gdir = RUN_DIR / "gateway"
    cfg_path = gdir / "litellm.yaml"
    sess = "lms-gateway"
    if a.gw_cmd in ("config", "up", "reload"):
        cfg, notes = gateway_config()
        gdir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("# generated by `lms gateway`; do not edit\n" + yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {cfg_path} with {len(cfg['model_list'])} backend(s):")
        for n in notes:
            print(f"  {n}")
        if a.gw_cmd == "config":
            return
    if a.gw_cmd in ("down", "reload"):
        stop_session(sess, grace=20)
        (REGISTRY_DIR / f"gateway-{HOST}.json").unlink(missing_ok=True)
        if a.gw_cmd == "down":
            print("gateway stopped")
            return
    if a.gw_cmd in ("up", "reload"):
        if session_exists(sess):
            print(f"{sess} already running; use `lms gateway reload` to pick up registry changes")
            return
        litellm = ROOT / ".venv" / "bin" / "litellm"
        if not litellm.exists():
            die("gateway env missing; run scripts/bootstrap.sh --gateway")
        for path, prefix in ((BACKEND_KEY, "lms-backend-"), (MASTER_KEY, "sk-lms-")):
            ensure_key(path, prefix)
        start_session(sess, [str(ROOT / "bin" / "lms"), "_gateway"])
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        (REGISTRY_DIR / f"gateway-{HOST}.json").write_text(json.dumps(
            {"kind": "gateway", "host": HOST, "url": f"http://{ADVERTISE}:{GATEWAY_PORT}", "ts": now_iso()}, indent=2) + "\n")
        print(f"{sess}: started on 0.0.0.0:{GATEWAY_PORT}; log {gdir / 'gateway.log'}")
    if a.gw_cmd == "status":
        print(f"{sess}: {'up' if session_exists(sess) else 'down'}")
        try:
            st, body = http_get(f"http://127.0.0.1:{GATEWAY_PORT}/v1/models", read_secret(MASTER_KEY), 5)
            print("models:", ", ".join(sorted({m["id"] for m in json.loads(body)["data"]})))
        except Exception as e:  # noqa: BLE001
            print(f"/v1/models: {type(e).__name__}: {e}")


def cmd_gateway_run(_a) -> None:
    """Internal: runs inside tmux. Load secrets into env (never argv) and exec LiteLLM."""
    gdir = RUN_DIR / "gateway"
    os.environ["LITELLM_MASTER_KEY"] = read_secret(MASTER_KEY)
    os.environ["LMS_BACKEND_KEY"] = read_secret(BACKEND_KEY)
    for k, v in read_env_file(CF_ACCESS_ENV).items():
        os.environ[k] = v
    cfg = yaml.safe_load((gdir / "litellm.yaml").read_text())
    # LiteLLM only resolves os.environ/ at the top level of litellm_params, so resolve
    # nested header refs into a private runtime copy (0600, gitignored run dir).
    resolved = gdir / "litellm.runtime.yaml"
    private_write(resolved, yaml.safe_dump(resolve_env_refs_nested_only(cfg), sort_keys=False))
    os.environ.setdefault("LITELLM_LOG", "INFO")
    os.environ["LITELLM_TELEMETRY"] = "False"
    os.environ.setdefault("DISABLE_ADMIN_UI", "True")
    logf = open(gdir / "gateway.log", "a")
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    print(f"==== {now_iso()} starting litellm on port {GATEWAY_PORT}", flush=True)
    os.execv(str(ROOT / ".venv" / "bin" / "litellm"),
             ["litellm", "--config", str(resolved), "--host", "0.0.0.0", "--port", str(GATEWAY_PORT),
              "--num_workers", os.environ.get("LMS_GATEWAY_WORKERS", "1")])


def resolve_env_refs_nested_only(cfg: dict) -> dict:
    for m in cfg.get("model_list", []):
        p = m.get("litellm_params", {})
        if "extra_headers" in p:
            p["extra_headers"] = resolve_env_refs(p["extra_headers"])
    return cfg


# --------------------------------------------------------------------------- tunnel (this machine's own)

def cmd_tunnel(a) -> None:
    sess = "lms-tunnel"
    if a.tun_cmd == "status":
        print(f"{sess}: {'up' if session_exists(sess) else 'down'}; token file {TUNNEL_TOKEN} "
              f"{'present' if TUNNEL_TOKEN.exists() else 'MISSING'}")
        return
    if a.tun_cmd == "down":
        stop_session(sess, grace=15)
        print(f"{sess}: stopped")
        return
    # up
    if session_exists(sess):
        die(f"{sess} already running")
    read_secret(TUNNEL_TOKEN)  # existence + 0600 check; value never leaves the file
    guard = Path.home() / ".config" / "slurm-dash" / "tunnel-token"
    if guard.exists() and guard.resolve() == TUNNEL_TOKEN.resolve():
        die("refusing: tunnel-token points at the Oscar dashboard tunnel token; each machine needs its own")
    if guard.exists() and guard.read_bytes().strip() == TUNNEL_TOKEN.read_bytes().strip():
        die("refusing: tunnel-token is identical to the Oscar dashboard tunnel token; each machine needs its own")
    cf = shutil.which("cloudflared") or str(Path.home() / ".local/bin/cloudflared")
    if not Path(cf).exists():
        die("cloudflared not found (install to ~/.local/bin; see README)")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # TUNNEL_TOKEN_FILE keeps the token out of argv (cloudflared >= 2025.4).
    tmux("new-session", "-d", "-s", sess, "-e", f"TUNNEL_TOKEN_FILE={TUNNEL_TOKEN}",
         f"exec {shlex.quote(cf)} tunnel --no-autoupdate run 2>&1 | tee -a {shlex.quote(str(RUN_DIR / 'tunnel.log'))}",
         check=True)
    print(f"{sess}: started")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(prog="lms", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("up", help="start a profile's instances (or `auto`)")
    p.add_argument("profile")
    p.add_argument("--only", nargs="+", help="start only these instances")
    p.add_argument("--dry-run", action="store_true", help="print commands, start nothing")
    p.add_argument("--fake-gpus", help="fake GPU inventory for auto/dry-run: N or NxMIB (e.g. 2x24000)")
    p.add_argument("--fake-ram-gib", type=float, help="fake host RAM for auto selection")
    p.set_defaults(fn=cmd_up)

    p = sub.add_parser("down", help="stop instances on this node (all if none given)")
    p.add_argument("instances", nargs="*")
    p.set_defaults(fn=cmd_down)

    sub.add_parser("status", help="show node, registry, gateway").set_defaults(fn=cmd_status)

    p = sub.add_parser("logs", help="show an instance log")
    p.add_argument("instance")
    p.add_argument("-n", "--lines", type=int, default=100)
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("fetch", help="download a profile's model files with `hf download`")
    p.add_argument("profile")
    p.add_argument("--only", nargs="+")
    p.add_argument("--tmux", action="store_true", help="run each download in tmux session lms-dl-<instance>")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fetch)

    sub.add_parser("profiles", help="list profiles").set_defaults(fn=lambda a: print("\n".join(list_profiles())))

    p = sub.add_parser("auth", help="keys and client auth.env")
    asub = p.add_subparsers(dest="auth_cmd", required=True)
    pi = asub.add_parser("init", help="generate backend/gateway keys and auth.env (0600)")
    pi.add_argument("--base-url", default=os.environ.get("LMS_PUBLIC_URL", "https://llm.garylvov.com"))
    pi.add_argument("--force", action="store_true")
    pc = asub.add_parser("print-client", help="emit an auth.env for another machine (pipe it, do not display)")
    pc.add_argument("--host", help="label for the target machine")
    pc.add_argument("--stdout", action="store_true", help="allow printing to a terminal")
    p.set_defaults(fn=cmd_auth)

    p = sub.add_parser("gateway", help="LiteLLM gateway from the registry")
    p.add_argument("gw_cmd", choices=["up", "reload", "down", "status", "config"])
    p.set_defaults(fn=cmd_gateway)

    p = sub.add_parser("tunnel", help="this machine's OWN cloudflared connector (tmux lms-tunnel)")
    p.add_argument("tun_cmd", choices=["up", "down", "status"])
    p.set_defaults(fn=cmd_tunnel)

    p = sub.add_parser("_serve")
    p.add_argument("profile")
    p.add_argument("instance")
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("_gateway").set_defaults(fn=cmd_gateway_run)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
