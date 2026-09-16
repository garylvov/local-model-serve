"""llm gateway: one public endpoint in front of many llama-server routers ("peers").

Peers heartbeat POST /peers/register {"url": ..., "cf_access": bool} every ~30 s (bin/llm does it).
Every POLL s the gateway reads each peer's /models. Requests are routed by the JSON "model"
field (or ?model=) to a peer with that model loaded (sticky per session header, else fewest
in-flight); else to a peer that lists it (router autoload); else a JSON 404/503.
Responses stream through byte for byte. Same shared API key for clients, peers and routers.
"""
import asyncio, base64, hashlib, hmac, json, os, secrets, time, uuid
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

CONF = Path(os.environ.get("LLM_CONFIG_DIR", Path.home() / ".config/local-model-serve"))
KEY = (CONF / "api-key").read_text().strip()
POLL, TTL, PIN_TTL = float(os.environ.get("LLM_GW_POLL", 10)), float(os.environ.get("LLM_GW_PEER_TTL", 90)), 7200
PASSWD = CONF / "passwd"          # scrypt hash written by `llm passwd`
COOKIE, COOKIE_TTL = "llm_session", 12 * 3600
LOGIN_WINDOW, LOGIN_MAX = 300, 5  # failed logins per IP per window
MUTATE_WINDOW, MUTATE_MAX = 60, 20  # model-mutating requests per IP per window (Models tab)
SESSION_HEADERS = ("x-session-id", "x-litellm-session-id", "x-claude-code-session-id", "session-id", "conversation-id")
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "authorization", "x-api-key", "accept-encoding", "proxy-authorization",
       # llama-server tool-execution overrides: never let internet clients pick a cwd or a runtime
       # (e.g. "ssh:<host>" / "docker-container:<id>") for POST /tools
       "x-tool-cwd", "x-tool-runtime", "x-resp-type"}

CATALOG_PATH = Path(os.environ.get("LLM_CATALOG", Path(__file__).resolve().parent.parent / "catalog/models.yaml"))


def load_catalog() -> dict:
    try:
        return (yaml.safe_load(CATALOG_PATH.read_text()) or {}).get("models", {})
    except (OSError, yaml.YAMLError) as e:
        print(f"catalog: failed to load {CATALOG_PATH}: {e}", flush=True)
        return {}


CATALOG = load_catalog()  # re-read once at startup; the catalog changes rarely and is not secret


def catalog_repos() -> set:
    return {m["repo"] for m in CATALOG.values() if m.get("repo")}


def catalog_quant_gib(hf_repo: str, quant: str | None):
    for m in CATALOG.values():
        if m.get("repo") == hf_repo:
            for q in m.get("quants", []) or []:
                if q.get("name") == quant or (quant is None and len(m.get("quants", [])) == 1):
                    return q.get("gib")
    return None


def cf_access_headers() -> dict:
    """CF Access service token for off-cluster peers (agents never see it)."""
    env, f = {}, CONF / "cf-access.env"
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'\"")
    ids = env.get("CF_ACCESS_CLIENT_ID"), env.get("CF_ACCESS_CLIENT_SECRET")
    return {"CF-Access-Client-Id": ids[0], "CF-Access-Client-Secret": ids[1]} if all(ids) else {}


peers: dict = {}   # url -> {seen, cf_access, models: {id: status}, ok, inflight, preset, disk_free_gib, cmd_log}
pins: dict = {}    # (session, model) -> (url, ts)
pending_commands: dict = {}  # url -> [{"id", "action", "params"}] ; drained into the next heartbeat response
client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None), limits=httpx.Limits(max_connections=2000))
mutations: dict = {}  # ip -> [timestamps] ; rate limit for Models-tab writes


def client_ip_generic(req: Request) -> str:
    return req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "?")


def csrf_ok(req: Request) -> bool:
    """Same-origin check for mutating Models-tab requests. The login cookie is SameSite=lax, which
    still allows a top-level cross-site GET to carry it, so every state-changing request needs an
    explicit same-origin signal too. A request authenticated with the bearer API key (never a
    cookie) cannot be forged cross-site by a browser in the first place, so it is exempt."""
    if authorized(req):
        return True
    sfs = req.headers.get("sec-fetch-site")
    if sfs is not None:
        return sfs in ("same-origin", "none")
    origin = req.headers.get("origin")
    if origin:
        try:
            return httpx.URL(origin).host == (req.headers.get("host", "").split(":")[0])
        except Exception:
            return False
    return False  # no signal at all: refuse rather than guess


def rate_limited(req: Request) -> bool:
    ip, now = client_ip_generic(req), time.time()
    hits = [t for t in mutations.get(ip, []) if now - t < MUTATE_WINDOW]
    hits.append(now)
    mutations[ip] = hits
    return len(hits) > MUTATE_MAX


def audit(req: Request, action: str, **detail) -> None:
    ip = client_ip_generic(req)
    extra = " ".join(f"{k}={v}" for k, v in detail.items())
    print(f"mutate ip={ip} action={action} {extra}", flush=True)


async def guard(req: Request):
    """Common gate for every mutating Models-tab endpoint: auth, same-origin, rate limit."""
    if not browser_ok(req):
        return err(401, "invalid API key")
    if not csrf_ok(req):
        return err(403, "cross-site request refused")
    if rate_limited(req):
        return err(429, "too many changes; slow down")
    return None


def peer_headers(p: dict) -> dict:
    h = {"Authorization": f"Bearer {KEY}", "x-api-key": KEY}
    return {**h, **cf_access_headers()} if p["cf_access"] else h


def authorized(req: Request) -> bool:
    auth = req.headers.get("authorization", "")
    k = auth[7:].strip() if auth.lower().startswith("bearer ") else req.headers.get("x-api-key", "")
    return bool(k) and hmac.compare_digest(k.encode(), KEY.encode())


def err(status: int, msg: str, anthropic: bool = False) -> JSONResponse:
    body = ({"type": "error", "error": {"type": "not_found_error" if status == 404 else "api_error", "message": msg}}
            if anthropic else {"error": {"message": msg, "type": "invalid_request_error" if status == 404 else "api_error", "code": status}})
    return JSONResponse(body, status_code=status)


def sign(payload: str) -> str:
    return hmac.new(KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_cookie() -> str:
    exp = str(int(time.time() + COOKIE_TTL))
    return f"{exp}.{sign(exp)}"


def cookie_ok(val: str | None) -> bool:
    try:
        exp, sig = (val or "").split(".", 1)
        return hmac.compare_digest(sig, sign(exp)) and time.time() < float(exp)
    except ValueError:
        return False


def password_ok(pw: str) -> bool:
    if not PASSWD.exists():
        return False
    salt, want = PASSWD.read_text().split()[:2]
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32).hex()
    return hmac.compare_digest(got, want)


logins: dict = {}  # ip -> [timestamps of failures]


def client_ip(req: Request) -> str:
    return req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "?")


def browser_ok(req: Request) -> bool:
    return authorized(req) or cookie_ok(req.cookies.get(COOKIE))


_dns: dict = {}  # host -> (ip or None, ts)


async def resolve(host: str):
    """Oscar's resolver returns nothing for *.trycloudflare.com (measured 2026-09-16), so when the
    system resolver fails, ask Cloudflare's DNS-over-HTTPS by IP. Returns an IP to dial, or None."""
    hit = _dns.get(host)
    if hit and time.time() - hit[1] < 300:
        return hit[0]
    ip = None
    try:
        await asyncio.get_running_loop().getaddrinfo(host, 443)
    except OSError:
        try:
            r = await client.get("https://1.1.1.1/dns-query", params={"name": host, "type": "A"},
                                 headers={"accept": "application/dns-json"}, timeout=5)
            ip = next((a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1), None)
        except (httpx.HTTPError, ValueError):
            pass
    _dns[host] = (ip, time.time())
    return ip


async def target(base: str, path: str):
    """(url, extra headers, httpx extensions) - dial a DoH-resolved IP but keep Host + TLS SNI."""
    u = httpx.URL(base + path)
    ip = await resolve(u.host) if u.scheme == "https" else None
    if not ip:
        return u, {}, {}
    return u.copy_with(host=ip), {"Host": u.host}, {"sni_hostname": u.host}


async def poll(url: str) -> None:
    p = peers.get(url)
    if not p:
        return
    try:
        u, h, ext = await target(url, "/models")
        r = await client.get(u, headers=peer_headers(p) | h, timeout=8, extensions=ext)
        r.raise_for_status()
        data = r.json().get("data", [])
        p["models"] = {m["id"]: m.get("status", {}).get("value", "loaded") for m in data}
        p["detail"] = {m["id"]: await model_detail(url, p, m) for m in data
                       if m.get("status", {}).get("value") in ("loaded", "loading", "sleeping")}
        p["ok"] = True
    except (httpx.HTTPError, ValueError) as e:
        p["ok"] = False
        print(f"poll {url}: {type(e).__name__}", flush=True)


def _arg(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


async def model_detail(url: str, p: dict, m: dict) -> dict:
    """GPU placement, speed, load and context use for one model on a peer. Only numbers and GPU
    indices are kept: never router args verbatim (they contain file paths and key-file locations)."""
    st = m.get("status", {})
    args = [str(a) for a in st.get("args", [])]
    dev = _arg(args, "--device") or ""
    gpus = [int(d[4:]) for d in dev.split(",") if d.startswith("CUDA") and d[4:].isdigit()]
    out = {"state": st.get("value"), "gpus": gpus, "parallel": int(_arg(args, "--parallel") or 0) or None}
    if st.get("value") != "loaded":
        return out
    try:
        u, h, ext = await target(url, "/metrics")
        r = await client.get(u, params={"model": m["id"]}, headers=peer_headers(p) | h, timeout=5, extensions=ext)
        met = {k: float(v) for k, _, v in (ln.partition(" ") for ln in r.text.splitlines()
                                            if ln.startswith("llamacpp:")) if v}
        u, h, ext = await target(url, "/slots")
        slots = (await client.get(u, params={"model": m["id"]}, headers=peer_headers(p) | h, timeout=5,
                                  extensions=ext)).json()
    except (httpx.HTTPError, ValueError):
        return out
    # tokens_predicted_total only grows when a request finishes, and the *_seconds gauges are reset
    # by each /metrics scrape, so report (a) throughput over a sliding ~60 s window and (b) the last
    # non-zero decode speed llama.cpp measured.
    now, total = time.time(), met.get("llamacpp:tokens_predicted_total", 0.0)
    hist = [x for x in p.setdefault("_tok", {}).get(m["id"], []) if now - x[0] <= 65 and x[1] <= total]
    hist.append((now, total))
    p["_tok"][m["id"]] = hist
    speed = met.get("llamacpp:predicted_tokens_seconds", 0.0)
    if speed > 0:
        p.setdefault("_speed", {})[m["id"]] = speed
    n_ctx = max((s.get("n_ctx", 0) for s in slots), default=0) if isinstance(slots, list) else 0
    out.update(
        tok_s=round((total - hist[0][1]) / (now - hist[0][0]), 1) if now - hist[0][0] >= 15 else None,
        avg_tok_s=round(p.get("_speed", {}).get(m["id"], 0.0), 1) or None,
        inflight=int(met.get("llamacpp:requests_processing", 0)), queued=int(met.get("llamacpp:requests_deferred", 0)),
        slots=len(slots) if isinstance(slots, list) else None,
        slots_busy=sum(1 for s in slots if s.get("is_processing")) if isinstance(slots, list) else None,
        # this llama.cpp build exports no live KV-usage metric; the peak context any request reached
        # (n_tokens_max) over the per-slot context is the closest honest number
        ctx_peak=int(met.get("llamacpp:n_tokens_max", 0)), ctx_slot=n_ctx or None)
    return out


async def poller() -> None:
    while True:
        now = time.time()
        for url in [u for u, p in peers.items() if now - p["seen"] > TTL]:
            print(f"peer {url} missed heartbeats; dropped", flush=True)
            peers.pop(url, None)
        for k in [k for k, (_, ts) in pins.items() if now - ts > PIN_TTL]:
            pins.pop(k, None)
        await asyncio.gather(*(poll(u) for u in list(peers)))
        await asyncio.sleep(POLL)


async def register(req: Request):
    if not authorized(req):
        return err(401, "invalid API key")
    body = await req.json()
    url = str(body.get("url", "")).rstrip("/")
    if not url.startswith(("http://", "https://")):
        return err(400, "url required")
    if req.url.path.endswith("/deregister"):
        peers.pop(url, None)
        return JSONResponse({"ok": True, "peers": len(peers)})
    new = url not in peers
    p = peers.setdefault(url, {"models": {}, "ok": False, "inflight": 0, "gpus": [], "host": url,
                               "preset": {}, "disk_free_gib": None, "cmd_log": []})
    p.update(seen=time.time(), cf_access=bool(body.get("cf_access")),
             gpus=body.get("gpus") or p.get("gpus") or [], host=body.get("host", url),
             preset=body.get("preset") or p.get("preset") or {},
             disk_free_gib=body.get("disk_free_gib", p.get("disk_free_gib")))
    for r in (body.get("results") or [])[:20]:
        entry = {"ts": time.time(), "id": r.get("id"), "ok": bool(r.get("ok")), "msg": str(r.get("msg", ""))[:500]}
        p["cmd_log"] = (p["cmd_log"] + [entry])[-20:]
        print(f"command result peer={url} id={entry['id']} ok={entry['ok']} msg={entry['msg']}", flush=True)
    if new:
        print(f"peer {url} registered", flush=True)
        await poll(url)
    cmds = pending_commands.pop(url, [])
    return JSONResponse({"ok": True, "peers": len(peers), "models": p["models"], "commands": cmds})


async def list_peers(req: Request):
    if not authorized(req):
        return err(401, "invalid API key")
    now = time.time()
    return JSONResponse({u: {"ok": p["ok"], "age_s": round(now - p["seen"]), "inflight": p["inflight"],
                             "models": p["models"]} for u, p in peers.items()})


async def models(req: Request):
    if not browser_ok(req):
        return err(401, "invalid API key")
    loaded = sorted({m for p in peers.values() if p["ok"] for m, s in p["models"].items() if s == "loaded"})
    # router-style "status" lets llama.cpp's WebUI (props role=router) list the models as usable
    return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "llm", "aliases": [],
                                                      "tags": [], "status": {"value": "loaded"}} for m in loaded]})


def peer_by_host(host: str):
    for u, p in peers.items():
        if p.get("host") == host or u == host:
            return u, p
    return None, None


async def machines(req: Request):
    """Models tab data source: every peer, its live model states/detail (from poll()) merged with
    what its preset declares (from the heartbeat), GPUs, free disk and the last command results."""
    if not browser_ok(req):
        return err(401, "invalid API key")
    now = time.time()
    out = []
    for u, p in sorted(peers.items()):
        models_by_id = {m["id"] if isinstance(m, dict) else m: s for m, s in p["models"].items()} if False else p["models"]
        merged = {}
        for name, cfg in (p.get("preset") or {}).items():
            merged[name] = {**cfg, "state": p["models"].get(name, "unloaded"), **p.get("detail", {}).get(name, {})}
        for name, state in p["models"].items():  # cached-but-not-in-preset models still show up
            if name not in merged:
                merged[name] = {"state": state, **p.get("detail", {}).get(name, {})}
        out.append({"host": p.get("host", u), "ok": p["ok"], "stale": now - p["seen"] > 45 or not p["ok"],
                    "age_s": round(now - p["seen"]), "gpus": p.get("gpus", []),
                    "disk_free_gib": p.get("disk_free_gib"), "models": merged,
                    "cmd_log": p.get("cmd_log", [])})
    return JSONResponse({"machines": out, "catalog_repos": sorted(catalog_repos())})


async def peer_call(p: dict, url: str, method: str, path: str, params=None, json_body=None):
    u, h, ext = await target(url, path)
    r = await client.request(method, u, params=params, json=json_body, headers=peer_headers(p) | h,
                             timeout=20, extensions=ext)
    return r


async def model_action(req: Request):
    """POST /machines/<host>/models/<model>/load|unload -- immediate, via the router API (no
    preset edit needed: load/unload just start/stop a child process)."""
    if (g := await guard(req)) is not None:
        return g
    host, model, action = req.path_params["host"], req.path_params["model"], req.path_params["action"]
    if action not in ("load", "unload"):
        return err(404, "unknown action")
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    if action == "unload":
        inflight = (p.get("detail", {}).get(model) or {}).get("inflight") or 0
        body = {}
        try:
            body = json.loads(await req.body() or b"{}")
        except ValueError:
            pass
        if inflight and not body.get("confirm"):
            return err(409, f"{inflight} request(s) in flight for {model}; pass confirm:true to unload anyway")
    try:
        r = await peer_call(p, url, "POST", f"/models/{action}", json_body={"model": model})
    except httpx.HTTPError as e:
        return err(502, f"peer unreachable: {type(e).__name__}")
    audit(req, f"model_{action}", host=host, model=model, status=r.status_code)
    await poll(url)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


async def model_download(req: Request):
    """POST /machines/<host>/models -- start a download via the router's own downloader.
    Body: {"repo": "<org>/<name>", "quant": "<TAG>"?, "custom_repo": bool?}. Refused unless the
    repo is in catalog/models.yaml, or the operator explicitly opts in with custom_repo:true
    (still logged). Capped by the peer's last-reported free disk."""
    if (g := await guard(req)) is not None:
        return g
    host = req.path_params["host"]
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    try:
        body = json.loads(await req.body() or b"{}")
    except ValueError:
        return err(400, "body must be JSON")
    repo, quant, custom = str(body.get("repo", "")).strip(), body.get("quant"), bool(body.get("custom_repo"))
    if not repo or "/" not in repo:
        return err(400, "repo required, e.g. org/name")
    known = repo in catalog_repos()
    if not known and not custom:
        return err(400, f"{repo!r} is not in catalog/models.yaml; pass custom_repo:true to override")
    size_gib = catalog_quant_gib(repo, quant)
    free = p.get("disk_free_gib")
    if size_gib is not None and free is not None and size_gib * 1.05 > free:
        return err(400, f"needs ~{size_gib:.1f} GiB, only {free:.1f} GiB free on {host}")
    hf_repo = f"{repo}:{quant}" if quant else repo
    try:
        r = await peer_call(p, url, "POST", "/models", json_body={"model": hf_repo})
    except httpx.HTTPError as e:
        return err(502, f"peer unreachable: {type(e).__name__}")
    audit(req, "model_download", host=host, repo=hf_repo, custom_repo=custom, status=r.status_code)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


async def model_delete(req: Request):
    if (g := await guard(req)) is not None:
        return g
    host, model = req.path_params["host"], req.path_params["model"]
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    try:
        body = json.loads(await req.body() or b"{}")
    except ValueError:
        body = {}
    if not body.get("confirm"):
        return err(409, "delete is destructive; pass confirm:true")
    try:
        r = await peer_call(p, url, "DELETE", "/models", params={"model": model})
    except httpx.HTTPError as e:
        return err(502, f"peer unreachable: {type(e).__name__}")
    audit(req, "model_delete", host=host, model=model, status=r.status_code)
    await poll(url)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))


async def model_sse(req: Request):
    """Proxy the router's live /models/sse for download/load progress. Falls back to polling in
    the UI if this 502s (peer unreachable / SSE not supported by this build)."""
    if not browser_ok(req):
        return err(401, "invalid API key")
    host = req.path_params["host"]
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    try:
        u, h, ext = await target(url, "/models/sse")
        up = await client.send(client.build_request("GET", u, headers=peer_headers(p) | h, extensions=ext),
                               stream=True)
    except httpx.HTTPError as e:
        return err(502, f"peer unreachable: {type(e).__name__}")

    async def relay():
        try:
            async for chunk in up.aiter_raw():
                yield chunk
        finally:
            await up.aclose()
    return StreamingResponse(relay(), status_code=up.status_code, media_type="text/event-stream")


# ---- placement (preset) editing: queued as a pull command, applied by bin/llm on its next beat --
ALLOWED_PATCH_KEYS = {"device", "ctx-size", "parallel", "no-mmproj", "threads", "hf-repo", "quant",
                      "image-min-tokens", "image-max-tokens", "n-gpu-layers"}


def vram_fit_warning(p: dict, gpu_idxs: list, hf_repo: str, quant) -> str | None:
    gpu_by_idx = {g["index"]: g for g in p.get("gpus", [])}
    missing = [i for i in gpu_idxs if i not in gpu_by_idx]
    if missing:
        return f"GPU(s) {missing} do not exist on {p.get('host')} (has {sorted(gpu_by_idx)})"
    size_gib = catalog_quant_gib(hf_repo, quant)
    if size_gib is None or not gpu_idxs:
        return None
    total_mib = sum(gpu_by_idx[i]["mem_total"] for i in gpu_idxs)
    need_mib = size_gib * 1024 * 1.12  # ~12% headroom for KV cache + compute buffers (README budget)
    if need_mib > total_mib:
        return (f"{hf_repo}:{quant} needs ~{need_mib/1024:.1f} GiB (incl. KV/compute headroom) "
                f"but GPU(s) {gpu_idxs} only have {total_mib/1024:.1f} GiB total")
    return None


async def set_preset(req: Request):
    if (g := await guard(req)) is not None:
        return g
    host = req.path_params["host"]
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    try:
        body = json.loads(await req.body() or b"{}")
    except ValueError:
        return err(400, "body must be JSON")
    section, patch = body.get("model"), body.get("patch")
    if not section or not isinstance(patch, dict) or not patch:
        return err(400, "body must be {model, patch:{...}}")
    bad = set(patch) - ALLOWED_PATCH_KEYS
    if bad:
        return err(400, f"unknown key(s): {sorted(bad)}")
    cfg = (p.get("preset") or {}).get(section, {})
    patch = dict(patch)
    if "quant" in patch:   # UI-only field: the INI encodes quant as a ":TAG" suffix on hf-repo
        base = str(patch.get("hf-repo") or cfg.get("hf_repo") or "").split(":", 1)[0]
        if not base:
            return err(400, "quant given without a known hf-repo for this model")
        patch["hf-repo"] = f"{base}:{patch.pop('quant')}" if patch["quant"] else base
        patch.pop("quant", None)
    gpu_idxs = None
    if "device" in patch:
        val = str(patch["device"])
        try:
            gpu_idxs = [int(t[4:]) for t in val.split(",") if t] if val else []
        except ValueError:
            return err(400, f"bad device value: {val!r}")
    hf_repo_full = str(patch.get("hf-repo") or (f"{cfg['hf_repo']}:{cfg['quant']}" if cfg.get("hf_repo") and cfg.get("quant")
                                                 else cfg.get("hf_repo") or ""))
    hf_repo, _, quant = hf_repo_full.partition(":")
    warn = vram_fit_warning(p, gpu_idxs if gpu_idxs is not None else (cfg.get("gpus") or []), hf_repo, quant or None) \
        if hf_repo else None
    if warn and not body.get("force"):
        return err(400, f"placement rejected: {warn}")
    cid = uuid.uuid4().hex[:12]
    pending_commands.setdefault(url, []).append(
        {"id": cid, "action": "set_preset", "params": {"section": section, "patch": {k: str(v) for k, v in patch.items()}}})
    audit(req, "set_preset", host=host, model=section, patch=patch, warn=bool(warn))
    return JSONResponse({"ok": True, "queued": cid, "applies_within_s": 30, "warning": warn})


async def restart_router(req: Request):
    if (g := await guard(req)) is not None:
        return g
    host = req.path_params["host"]
    url, p = peer_by_host(host)
    if not p:
        return err(404, f"unknown machine {host!r}")
    try:
        body = json.loads(await req.body() or b"{}")
    except ValueError:
        body = {}
    any_loaded = any(v == "loaded" for v in p["models"].values())
    if any_loaded and not body.get("confirm"):
        return err(409, "restarting will drop every loaded model on this machine; pass confirm:true")
    cid = uuid.uuid4().hex[:12]
    pending_commands.setdefault(url, []).append({"id": cid, "action": "restart_router", "params": {}})
    audit(req, "restart_router", host=host)
    return JSONResponse({"ok": True, "queued": cid, "applies_within_s": 30})


def session_key(req: Request, body: dict):
    for h in SESSION_HEADERS:
        if req.headers.get(h):
            return req.headers[h]
    uid = (body.get("metadata") or {}).get("user_id") if isinstance(body.get("metadata"), dict) else None
    return uid or body.get("user")


def pick(model: str, sess):
    ready = [u for u, p in peers.items() if p["ok"] and p["models"].get(model) == "loaded"]
    if sess and (sess, model) in pins and pins[(sess, model)][0] in ready:
        url = pins[(sess, model)][0]
    elif ready:
        url = min(ready, key=lambda u: peers[u]["inflight"])
    else:
        loadable = [u for u, p in peers.items() if p["ok"] and model in p["models"]]
        if not loadable:
            return None, False
        url = min(loadable, key=lambda u: sum(s == "loaded" for s in peers[u]["models"].values()))
    if sess:
        pins[(sess, model)] = (url, time.time())
    return url, url not in ready


def default_peer():
    ok = [u for u, p in peers.items() if p["ok"]]
    local = [u for u in ok if "127.0.0.1" in u or "localhost" in u or os.uname().nodename.split(".")[0] in u]
    return (local or ok or [None])[0]


async def proxy(req: Request):
    path, anthropic = req.url.path, "/messages" in req.url.path
    api_path = path.startswith(("/v1/", "/chat/", "/completions", "/infill", "/apply-template", "/tokenize"))
    if api_path and not browser_ok(req):   # key, or the WebUI's login cookie (SameSite=lax blocks cross-site POSTs)
        return err(401, "invalid API key", anthropic)
    if not api_path and not browser_ok(req):   # WebUI and everything else: login or key
        return RedirectResponse("/login", status_code=303) if req.method == "GET" else err(401, "login required")
    raw = await req.body()
    body = {}
    if raw:
        try:
            body = json.loads(raw)
        except ValueError:
            return err(400, "request body must be JSON", anthropic)
    model = body.get("model") if isinstance(body, dict) else None
    model = model or req.query_params.get("model")
    if not model:   # WebUI assets, /props, /health, ... : any healthy peer (this machine first)
        url, autoload = default_peer(), False
        if not url:
            return err(503, "no peer registered", anthropic)
        return await forward(req, url, raw, autoload, anthropic)
    url, autoload = pick(model, session_key(req, body))
    if not url:
        known = any(model in p["models"] for p in peers.values())
        return err(503 if known else 404, f"model '{model}' is {'on no healthy peer' if known else 'not served by any peer'}", anthropic)
    return await forward(req, url, raw, autoload, anthropic)


async def forward(req: Request, url: str, raw: bytes, autoload: bool, anthropic: bool):
    p = peers[url]
    # Browser (WebUI) paths keep the client's Accept-Encoding so llama.cpp can serve its
    # pre-compressed assets; API paths stay uncompressed so SSE streams are never buffered.
    hop = HOP - {"accept-encoding"} if not req.url.path.startswith(("/v1/", "/chat/")) else HOP
    q = "&".join(x for x in (str(req.url.query), "autoload=true" if autoload else "") if x)
    headers = {k: v for k, v in req.headers.items() if k.lower() not in hop} | peer_headers(p)
    p["inflight"] += 1
    try:
        u, h, ext = await target(url, req.url.path)
        up = await client.send(client.build_request(req.method, u, params=q or None, headers=headers | h,
                                                    content=raw, extensions=ext), stream=True)
    except httpx.HTTPError as e:
        p["inflight"] -= 1
        p["ok"] = False
        return err(502, f"peer unreachable: {type(e).__name__}", anthropic)

    async def relay():
        try:
            async for chunk in up.aiter_raw():
                yield chunk
        finally:
            await up.aclose()
            p["inflight"] -= 1

    out = {k: v for k, v in up.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection")}
    if (req.method == "GET" and req.url.path in ("/", "/index.html") and up.status_code == 200
            and up.headers.get("content-type", "").startswith("text/html")):
        # Only the WebUI's small HTML shell is rewritten (to add the Chat | Hardware bar);
        # API, SSE and asset responses always stream through untouched.
        try:
            body = await up.aread()
        finally:
            await up.aclose()
            p["inflight"] -= 1
        # aread() already undoes Content-Encoding (llama.cpp ships the shell pre-gzipped)
        html = body.decode("utf-8", "replace")
        bar = TABBAR % (ACTIVE, "", "")
        html = html.replace("</body>", bar + "</body>", 1) if "</body>" in html else html + bar
        out = {k: v for k, v in out.items() if k.lower() not in ("content-encoding", "etag")}
        out["cache-control"] = "no-store"
        return Response(html, status_code=200, headers=out, media_type="text/html; charset=utf-8")
    return StreamingResponse(relay(), status_code=up.status_code, headers=out)


LOGIN_HTML = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>llm.garylvov.com</title><style>body{font:16px system-ui;background:#111;color:#eee;display:grid;
place-items:center;height:100vh;margin:0}form{display:grid;gap:.6rem;min-width:18rem}input,button{padding:.6rem;
font:inherit;border-radius:.4rem;border:1px solid #444;background:#1c1c1c;color:#eee}button{cursor:pointer}
.e{color:#f77}</style><form method=post action=/login><h1>local models</h1>
<input type=password name=password placeholder=password autofocus autocomplete=current-password>
<button>sign in</button>%s</form>"""


async def login(req: Request):
    if req.method == "GET":
        return HTMLResponse(LOGIN_HTML % "")
    ip, now = client_ip(req), time.time()
    fails = [t for t in logins.get(ip, []) if now - t < LOGIN_WINDOW]
    if len(fails) >= LOGIN_MAX:
        logins[ip] = fails
        print(f"login: rate-limited ip={ip}", flush=True)
        return HTMLResponse(LOGIN_HTML % "<p class=e>too many attempts, wait a few minutes</p>", status_code=429)
    form = parse_qs((await req.body()).decode())        # no python-multipart dependency
    if not password_ok((form.get("password") or [""])[0]):
        logins[ip] = fails + [now]
        print(f"login: failed ip={ip} attempts={len(fails) + 1}", flush=True)
        return HTMLResponse(LOGIN_HTML % "<p class=e>wrong password</p>", status_code=401)
    logins.pop(ip, None)
    print(f"login: ok ip={ip}", flush=True)
    r = RedirectResponse("/", status_code=303)
    r.set_cookie(COOKIE, make_cookie(), max_age=COOKIE_TTL, httponly=True, samesite="lax",
                 secure=req.headers.get("x-forwarded-proto", req.url.scheme) == "https")
    return r


def peer_kind(url: str) -> str:
    host = httpx.URL(url).host
    if host.endswith(".trycloudflare.com"):
        return "quick tunnel"
    return "home rig" if url.startswith("https://") else "Oscar"


def status_data() -> dict:
    """Existing keys (ts, peers[].host/url/ok/age_s/inflight/models/gpus) are unchanged; kind, stale and
    model_detail were added for the hardware tab. The page itself never renders url."""
    now = time.time()
    return {"ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "heartbeat_ttl_s": TTL, "peers": [
        {"host": p.get("host", u), "url": u, "ok": p["ok"], "age_s": round(now - p["seen"]), "inflight": p["inflight"],
         "models": p["models"], "gpus": p.get("gpus", []), "kind": peer_kind(u),
         "stale": now - p["seen"] > 45 or not p["ok"], "model_detail": p.get("detail", {})}
        for u, p in sorted(peers.items())]}


async def status_json(req: Request):
    if not browser_ok(req):
        return err(401, "login required")
    return JSONResponse(status_data())


TABBAR = ("<nav id=llm-tabs style=\"position:fixed;top:6px;left:50%%;transform:translateX(-50%%);z-index:2147483647;"
          "display:flex;gap:2px;padding:3px;border-radius:999px;font:600 12px system-ui,sans-serif;"
          "background:rgba(127,127,127,.22);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)\">"
          "<a href=/ style=\"padding:4px 12px;border-radius:999px;color:inherit;text-decoration:none;%s\">Chat</a>"
          "<a href=/status style=\"padding:4px 12px;border-radius:999px;color:inherit;text-decoration:none;%s\">Hardware</a>"
          "<a href=/models-ui style=\"padding:4px 12px;border-radius:999px;color:inherit;text-decoration:none;%s\">Models</a>"
          "</nav>")
ACTIVE = "background:rgba(127,127,127,.35)"

HARDWARE_HTML = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Hardware - llm.garylvov.com</title>
<meta name=color-scheme content="light dark"><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mute:#6b7280;--line:#e5e7eb;--track:#e9ecf0;--ok:#16a34a;
--warn:#d97706;--hot:#dc2626;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161b22;--fg:#e6edf3;--mute:#8b949e;--line:#262c36;
--track:#232a33;--ok:#3fb950;--warn:#d29922;--hot:#f85149;--acc:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;
padding:52px 16px 24px}main{max-width:1500px;margin:0 auto}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1rem;margin-bottom:14px}
h1{font-size:18px;margin:0}.mute{color:var(--mute)}.grid{display:grid;gap:18px;grid-template-columns:1fr}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card.stale{opacity:.45;filter:grayscale(1)}.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.host{font-weight:650;font-size:16px}.dot{width:9px;height:9px;border-radius:50%%;background:var(--ok)}
.stale .dot{background:var(--mute)}.pill{font-size:11px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);
color:var(--mute)}.sub{margin:2px 0 10px;font-size:12px}
/* wide: one row per GPU = id | name + model | util bar | VRAM bar | temp/power; narrow: bars stack full width */
.gpu{display:grid;grid-template-columns:2em minmax(10em,16em) minmax(0,1fr) minmax(0,1.4fr) 7.5em;gap:6px 16px;
align-items:center;padding:9px 0;border-top:1px solid var(--line)}.gi{font-weight:650;color:var(--mute)}
.gn{font-size:13px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gn .tag{display:block;overflow:hidden;text-overflow:ellipsis}.gt{font-size:13px;text-align:right;white-space:nowrap}
.bars{display:contents}.bar{position:relative;height:22px;border-radius:6px;background:var(--track);overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--acc);transition:width .4s}.bar b{position:relative;
font:600 12px/22px system-ui;padding-left:8px;white-space:nowrap}.bar.u i{background:var(--ok)}.bar.hi i{background:var(--warn)}
@media (max-width:760px){.gpu{grid-template-columns:2em 1fr auto}.gt{grid-column:3;grid-row:1}
.bars{display:grid;grid-column:1/-1;gap:5px}}
.bar.full i{background:var(--hot)}.tag{font-size:11px;color:var(--mute)}
.model{margin-top:10px;padding:8px 10px;border-radius:8px;background:var(--track)}
.mname{font-weight:650}.kv{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:12px;color:var(--mute)}
.kv b{color:var(--fg);font-weight:600}.empty{padding:30px;text-align:center}
</style>%(tabs)s<main><header><h1>Hardware</h1><span class=mute id=ts>loading...</span></header>
<div class=grid id=grid></div></main><script>
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct=(a,b)=>b?Math.max(0,Math.min(100,100*a/b)):0;
const cls=p=>p>=90?" full":p>=70?" hi":"";
function bar(p,label,extra){return `<div class="bar${extra}${cls(p)}"><i style="width:${p.toFixed(1)}%%"></i><b>${esc(label)}</b></div>`}
function ago(s){return s<60?s+"s":Math.floor(s/60)+"m "+(s%%60)+"s"}
function render(d){
  document.getElementById("ts").textContent="updated "+d.ts;
  const g=document.getElementById("grid");
  if(!d.peers.length){g.innerHTML='<div class="card empty mute">No machines are heartbeating.</div>';return}
  g.innerHTML=d.peers.map(p=>{
    const det=p.model_detail||{}, gm={};
    for(const[m,x] of Object.entries(det)) for(const i of x.gpus||[]) (gm[i]=gm[i]||[]).push(m);
    const gpus=(p.gpus||[]).map(x=>{const u=+x.util||0, mp=pct(+x.mem_used,+x.mem_total);
      return `<div class=gpu><span class=gi>${esc(x.index)}</span><span class=gn>${esc(x.name)}<span class=tag>${gm[x.index]?esc(gm[x.index].join(", ")):"idle"}</span></span>
      <div class=bars>${bar(u,`util ${u}%%`," u")}${bar(mp,`VRAM ${(x.mem_used/1024).toFixed(1)} / ${(x.mem_total/1024).toFixed(1)} GiB`,"")}</div>
      <span class=gt>${esc(x.temp)} °C · ${x.power==null?"-":Math.round(x.power)} W</span></div>`}).join("");
    const models=Object.entries(det).map(([m,x])=>{
      const ctx=x.ctx_slot?`${x.ctx_peak.toLocaleString()} / ${x.ctx_slot.toLocaleString()} (${pct(x.ctx_peak,x.ctx_slot).toFixed(0)}%%)`:"-";
      return `<div class=model><span class=mname>${esc(m)}</span> <span class=pill>${esc(x.state)}</span>
      <div class=kv><span>GPUs <b>${(x.gpus||[]).join(", ")||"-"}</b></span>
      <span>last 60 s <b>${x.tok_s==null?"-":x.tok_s} tok/s</b></span><span>decode <b>${x.avg_tok_s??"-"} tok/s</b></span>
      <span>in flight <b>${x.inflight??"-"}</b>${x.queued?` (+${x.queued} queued)`:""}</span>
      <span>slots busy <b>${x.slots_busy??"-"} / ${x.slots??"-"}</b></span><span>peak ctx/slot <b>${ctx}</b></span></div></div>`}).join("");
    return `<section class="card${p.stale?" stale":""}"><div class=top><span class=dot></span><span class=host>${esc(p.host)}</span>
      <span class=pill>${esc(p.kind)}</span><span class=pill>${p.stale?"offline":"online"}</span></div>
      <div class="sub mute">heartbeat ${ago(p.age_s)} ago | ${p.inflight} request(s) via gateway${p.stale?" | drops after "+d.heartbeat_ttl_s+"s":""}</div>
      ${gpus||'<div class=mute>no GPU data</div>'}${models||'<div class="model mute">no model loaded</div>'}</section>`}).join("");
}
async function tick(){try{const r=await fetch("/status.json",{credentials:"same-origin",cache:"no-store"});
  if(r.status===401){location.href="/login";return} render(await r.json())}
  catch(e){document.getElementById("ts").textContent="connection lost, retrying..."}
  setTimeout(tick,2000)}
tick();
</script></html>"""


async def status_page(req: Request):
    if not browser_ok(req):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(HARDWARE_HTML % {"tabs": TABBAR % ("", ACTIVE, "")}, headers={"cache-control": "no-store"})


MODELS_HTML = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Models - llm.garylvov.com</title>
<meta name=color-scheme content="light dark"><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mute:#6b7280;--line:#e5e7eb;--track:#e9ecf0;--ok:#16a34a;
--warn:#d97706;--hot:#dc2626;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161b22;--fg:#e6edf3;--mute:#8b949e;--line:#262c36;
--track:#232a33;--ok:#3fb950;--warn:#d29922;--hot:#f85149;--acc:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;
padding:52px 16px 24px}main{max-width:1100px;margin:0 auto}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:.4rem 1rem;margin-bottom:14px}
h1{font-size:18px;margin:0}.mute{color:var(--mute)}.grid{display:grid;gap:18px;grid-template-columns:1fr}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card.stale{opacity:.45;filter:grayscale(1)}.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.host{font-weight:650;font-size:16px}.pill{font-size:11px;padding:1px 8px;border-radius:999px;border:1px solid var(--line);
color:var(--mute)}.pill.loaded{color:var(--ok);border-color:var(--ok)}.pill.loading,.pill.downloading{color:var(--warn);border-color:var(--warn)}
.sub{margin:2px 0 10px;font-size:12px}
.model{margin-top:10px;padding:10px 12px;border-radius:8px;background:var(--track)}
.mrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.mname{font-weight:650}.kv{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:12px;color:var(--mute);margin-top:4px}
.kv b{color:var(--fg);font-weight:600}
button{font:600 12px system-ui;padding:5px 10px;border-radius:6px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}button:hover{border-color:var(--acc)}
button.danger:hover{border-color:var(--hot);color:var(--hot)}button:disabled{opacity:.5;cursor:default}
button.edit{margin-left:auto}
.form{display:none;margin-top:10px;padding:10px;border-top:1px dashed var(--line);gap:8px 14px;
grid-template-columns:repeat(auto-fit,minmax(110px,1fr))}
.form.open{display:grid}.form label{font-size:11px;color:var(--mute);display:block;margin-bottom:2px}
.form input,.form select{width:100%%;font:inherit;padding:4px 6px;border-radius:5px;border:1px solid var(--line);
background:var(--bg);color:var(--fg)}
.gpus{display:flex;flex-wrap:wrap;gap:4px}.gpus label{display:flex;align-items:center;gap:3px;font-size:11px;
border:1px solid var(--line);border-radius:5px;padding:2px 6px;cursor:pointer;margin:0}
.formrow{grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.err{color:var(--hot);font-size:12px}.warn{color:var(--warn);font-size:12px}
.dl{margin-top:10px;padding:10px;border-radius:8px;background:var(--track);display:grid;gap:6px}
.dl input{font:inherit;padding:5px 7px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
progress{width:100%%;height:8px}
.empty{padding:30px;text-align:center}
</style>%(tabs)s<main><header><h1>Models</h1><span class=mute id=ts>loading...</span></header>
<div class=grid id=grid></div></main>
<script>
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let DATA={machines:[]};
async function api(path,opts){
  const r=await fetch(path,{credentials:"same-origin",...opts});
  let body=null; try{body=await r.json()}catch(e){}
  if(!r.ok) throw new Error((body&&(body.error&&body.error.message||body.error))||(r.status+" "+r.statusText));
  return body;
}
function stateClass(s){return s==="loaded"?"loaded":(s==="loading"||s==="downloading")?"loading":""}
async function doLoad(host,model){
  const btn=event.target; btn.disabled=true;
  try{await api(`/machines/${encodeURIComponent(host)}/models/${encodeURIComponent(model)}/load`,{method:"POST"}); await tick()}
  catch(e){alert("load failed: "+e.message)} finally{btn.disabled=false}
}
async function doUnload(host,model,inflight){
  if(inflight>0 && !confirm(`${model} has ${inflight} request(s) in flight on ${host}. Unload anyway?`))return;
  const btn=event.target; btn.disabled=true;
  try{await api(`/machines/${encodeURIComponent(host)}/models/${encodeURIComponent(model)}/unload`,
    {method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({confirm:inflight>0})}); await tick()}
  catch(e){alert("unload failed: "+e.message)} finally{btn.disabled=false}
}
async function doDelete(host,model){
  if(!confirm(`Permanently delete the downloaded weights for ${model} on ${host}?`))return;
  try{await api(`/machines/${encodeURIComponent(host)}/models/${encodeURIComponent(model)}`,
    {method:"DELETE",headers:{"content-type":"application/json"},body:JSON.stringify({confirm:true})}); await tick()}
  catch(e){alert("delete failed: "+e.message)}
}
function toggleForm(id){const f=document.getElementById(id);f.classList.toggle("open")}
async function savePlacement(host,model,formId){
  const f=document.getElementById(formId);
  const gpus=[...f.querySelectorAll(".gpu-cb:checked")].map(c=>"CUDA"+c.value).join(",");
  const patch={device:gpus, "ctx-size":f.querySelector(".f-ctx").value, parallel:f.querySelector(".f-par").value,
    "no-mmproj":f.querySelector(".f-mmproj").checked?"false":"true"};
  const errEl=f.querySelector(".form-err"); errEl.textContent="";
  try{
    const r=await api(`/machines/${encodeURIComponent(host)}/preset`,{method:"POST",
      headers:{"content-type":"application/json"}, body:JSON.stringify({model, patch})});
    errEl.className="warn"; errEl.textContent=r.warning?("queued, but: "+r.warning):"queued - applies within ~30s (next heartbeat)";
  }catch(e){errEl.className="err"; errEl.textContent=e.message}
}
async function startDownload(host,formId){
  const f=document.getElementById(formId);
  const repo=f.querySelector(".dl-repo").value.trim(), quant=f.querySelector(".dl-quant").value.trim();
  const custom=f.querySelector(".dl-custom").checked;
  const errEl=f.querySelector(".dl-err"); errEl.textContent="";
  try{
    await api(`/machines/${encodeURIComponent(host)}/models`,{method:"POST",headers:{"content-type":"application/json"},
      body:JSON.stringify({repo, quant: quant||undefined, custom_repo: custom})});
    errEl.className="warn"; errEl.textContent="download started";
    await tick();
  }catch(e){errEl.className="err"; errEl.textContent=e.message}
}
function render(d){
  DATA=d;
  document.getElementById("ts").textContent="updated "+new Date().toLocaleTimeString();
  const g=document.getElementById("grid");
  if(!d.machines.length){g.innerHTML='<div class="card empty mute">No machines are heartbeating.</div>';return}
  g.innerHTML=d.machines.map((m,mi)=>{
    const gpuList=(m.gpus||[]).map(x=>x.index);
    const models=Object.entries(m.models||{}).map(([name,x],i)=>{
      const fid=`f-${mi}-${i}`;
      const gpus=x.gpus||[];
      const gpuCbs=gpuList.map(gi=>`<label><input type=checkbox class=gpu-cb value="${gi}" ${gpus.includes(gi)?"checked":""}>${gi}</label>`).join("");
      const loaded=x.state==="loaded";
      return `<div class=model>
        <div class=mrow><span class=mname>${esc(name)}</span><span class="pill ${stateClass(x.state)}">${esc(x.state||"unloaded")}</span>
        ${loaded?`<button class=danger onclick="doUnload('${esc(m.host)}','${esc(name)}',${x.inflight||0})">Unload</button>`
                :`<button onclick="doLoad('${esc(m.host)}','${esc(name)}')">Load</button>`}
        <button class=danger onclick="doDelete('${esc(m.host)}','${esc(name)}')">Delete weights</button>
        <button class=edit onclick="toggleForm('${fid}')">Edit placement</button></div>
        <div class=kv><span>GPUs <b>${gpus.join(", ")||"-"}</b></span>
        ${x.quant?`<span>quant <b>${esc(x.quant)}</b></span>`:""}
        ${x.ctx_size||x.ctx_slot?`<span>ctx <b>${(x.ctx_size||x.ctx_slot).toLocaleString()}</b></span>`:""}
        ${x.parallel?`<span>parallel <b>${x.parallel}</b></span>`:""}
        ${x.avg_tok_s?`<span>decode <b>${x.avg_tok_s} tok/s</b></span>`:""}
        ${x.inflight!=null?`<span>in flight <b>${x.inflight}</b></span>`:""}</div>
        <div class=form id="${fid}">
          <div><label>GPUs</label><div class=gpus>${gpuCbs||"<span class=mute>no GPU data</span>"}</div></div>
          <div><label>ctx-size</label><input class=f-ctx type=number value="${x.ctx_size||x.ctx_slot||131072}"></div>
          <div><label>parallel</label><input class=f-par type=number value="${x.parallel||4}"></div>
          <div><label>mmproj</label><label style="display:flex;align-items:center;gap:4px;border:none;padding:0">
            <input class=f-mmproj type=checkbox ${x.no_mmproj?"":"checked"}> enabled</label></div>
          <div class=formrow><button onclick="savePlacement('${esc(m.host)}','${esc(name)}','${fid}')">Save (writes preset, applies on next heartbeat)</button>
          <span class="form-err mute"></span></div>
        </div>
      </div>`}).join("") || '<div class="model mute">no models in this machine\\'s preset</div>';
    const dlid=`dl-${mi}`;
    return `<section class="card${m.stale?" stale":""}"><div class=top><span class=host>${esc(m.host)}</span>
      <span class=pill>${m.stale?"offline":"online"}</span>
      ${m.disk_free_gib!=null?`<span class=pill>${m.disk_free_gib.toFixed?m.disk_free_gib.toFixed(1):m.disk_free_gib} GiB free</span>`:""}</div>
      <div class="sub mute">${gpuList.length} GPU(s)</div>
      ${models}
      <div class=dl id="${dlid}"><div class=mrow><b>Download a model</b></div>
        <div class=mrow><input class=dl-repo placeholder="org/repo (from catalog)" style="flex:2;min-width:12em">
        <input class=dl-quant placeholder="quant tag e.g. UD-Q4_K_XL" style="flex:1;min-width:10em">
        <label style="display:flex;align-items:center;gap:4px;font-size:12px"><input type=checkbox class=dl-custom> custom repo</label>
        <button onclick="startDownload('${esc(m.host)}','${dlid}')">Start</button></div>
        <span class="dl-err mute"></span>
        <span class=mute style="font-size:11px">Catalog repos: ${esc((d.catalog_repos||[]).slice(0,6).join(", "))}${d.catalog_repos&&d.catalog_repos.length>6?"...":""}</span>
      </div>
      </section>`}).join("");
}
async function tick(){try{const r=await fetch("/machines",{credentials:"same-origin",cache:"no-store"});
  if(r.status===401){location.href="/login";return} render(await r.json())}
  catch(e){document.getElementById("ts").textContent="connection lost, retrying..."}
  setTimeout(tick,4000)}
tick();
</script></html>"""


async def models_page(req: Request):
    if not browser_ok(req):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(MODELS_HTML % {"tabs": TABBAR % ("", "", ACTIVE)}, headers={"cache-control": "no-store"})


async def health(_req: Request):
    return JSONResponse({"ok": True, "peers": len(peers)})


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(poller())
    yield
    task.cancel()


routes = Starlette(lifespan=lifespan, routes=[
    Route("/health", health), Route("/peers", list_peers),
    Route("/login", login, methods=["GET", "POST"]),
    Route("/status", status_page), Route("/status.json", status_json),
    Route("/peers/register", register, methods=["POST"]), Route("/peers/deregister", register, methods=["POST"]),
    Route("/v1/models", models), Route("/models", models),
    Route("/models-ui", models_page),
    Route("/machines", machines),
    Route("/machines/{host}/models/{model}/{action}", model_action, methods=["POST"]),
    Route("/machines/{host}/models/sse", model_sse),
    Route("/machines/{host}/models", model_download, methods=["POST"]),
    Route("/machines/{host}/models/{model}", model_delete, methods=["DELETE"]),
    Route("/machines/{host}/preset", set_preset, methods=["POST"]),
    Route("/machines/{host}/restart", restart_router, methods=["POST"]),
    Route("/{path:path}", proxy, methods=["GET", "POST"]),
])


async def app(scope, receive, send):
    """Pure ASGI (keeps streaming intact): via Cloudflare, force HTTPS and send HSTS.
    Internal http://<gateway>:4000 traffic carries no X-Forwarded-Proto and passes through untouched."""
    if scope["type"] != "http":
        return await routes(scope, receive, send)
    h = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
    proto = h.get("x-forwarded-proto") if "cf-connecting-ip" in h else None
    if proto == "http":
        qs = scope.get("query_string", b"").decode()
        loc = f"https://{h.get('host', '')}{scope['path']}" + (f"?{qs}" if qs else "")
        return await RedirectResponse(loc, status_code=308)(scope, receive, send)

    async def send_hsts(msg):
        if proto == "https" and msg["type"] == "http.response.start":
            msg["headers"] = list(msg.get("headers", [])) + [
                (b"strict-transport-security", b"max-age=31536000"), (b"x-frame-options", b"DENY"),
                (b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"same-origin")]
        await send(msg)
    return await routes(scope, receive, send_hsts)
