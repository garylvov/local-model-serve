"""llm gateway: one public endpoint in front of many llama-server routers ("peers").

Peers heartbeat POST /peers/register {"url": ..., "cf_access": bool} every ~30 s (bin/llm does it).
Every POLL s the gateway reads each peer's /models. Requests are routed by the JSON "model"
field (or ?model=) to a peer with that model loaded (sticky per session header, else fewest
in-flight); else to a peer that lists it (router autoload); else a JSON 404/503.
Responses stream through byte for byte. Same shared API key for clients, peers and routers.
"""
import asyncio, base64, hashlib, hmac, json, os, secrets, time
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from urllib.parse import parse_qs

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from starlette.routing import Route

CONF = Path(os.environ.get("LLM_CONFIG_DIR", Path.home() / ".config/local-model-serve"))
KEY = (CONF / "api-key").read_text().strip()
POLL, TTL, PIN_TTL = float(os.environ.get("LLM_GW_POLL", 10)), float(os.environ.get("LLM_GW_PEER_TTL", 90)), 7200
PASSWD = CONF / "passwd"          # scrypt hash written by `llm passwd`
COOKIE, COOKIE_TTL = "llm_session", 12 * 3600
LOGIN_WINDOW, LOGIN_MAX = 300, 5  # failed logins per IP per window
SESSION_HEADERS = ("x-session-id", "x-litellm-session-id", "x-claude-code-session-id", "session-id", "conversation-id")
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "authorization", "x-api-key", "accept-encoding", "proxy-authorization"}


def cf_access_headers() -> dict:
    """CF Access service token for off-cluster peers (agents never see it)."""
    env, f = {}, CONF / "cf-access.env"
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip("'\"")
    ids = env.get("CF_ACCESS_CLIENT_ID"), env.get("CF_ACCESS_CLIENT_SECRET")
    return {"CF-Access-Client-Id": ids[0], "CF-Access-Client-Secret": ids[1]} if all(ids) else {}


peers: dict = {}   # url -> {seen, cf_access, models: {id: status}, ok, inflight}
pins: dict = {}    # (session, model) -> (url, ts)
client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None), limits=httpx.Limits(max_connections=2000))


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
        p["models"] = {m["id"]: m.get("status", {}).get("value", "loaded") for m in r.json().get("data", [])}
        p["ok"] = True
    except (httpx.HTTPError, ValueError) as e:
        p["ok"] = False
        print(f"poll {url}: {type(e).__name__}", flush=True)


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
    p = peers.setdefault(url, {"models": {}, "ok": False, "inflight": 0, "gpus": [], "host": url})
    p.update(seen=time.time(), cf_access=bool(body.get("cf_access")),
             gpus=body.get("gpus") or p.get("gpus") or [], host=body.get("host", url))
    if new:
        print(f"peer {url} registered", flush=True)
        await poll(url)
    return JSONResponse({"ok": True, "peers": len(peers), "models": p["models"]})


async def list_peers(req: Request):
    if not authorized(req):
        return err(401, "invalid API key")
    now = time.time()
    return JSONResponse({u: {"ok": p["ok"], "age_s": round(now - p["seen"]), "inflight": p["inflight"],
                             "models": p["models"]} for u, p in peers.items()})


async def models(req: Request):
    if not authorized(req):
        return err(401, "invalid API key")
    loaded = sorted({m for p in peers.values() if p["ok"] for m, s in p["models"].items() if s == "loaded"})
    return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "owned_by": "llm"} for m in loaded]})


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
    if api_path and not authorized(req):
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


def status_data() -> dict:
    now = time.time()
    return {"ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "peers": [
        {"host": p.get("host", u), "url": u, "ok": p["ok"], "age_s": round(now - p["seen"]), "inflight": p["inflight"],
         "models": p["models"], "gpus": p.get("gpus", [])} for u, p in sorted(peers.items())]}


async def status_json(req: Request):
    if not browser_ok(req):
        return err(401, "login required")
    return JSONResponse(status_data())


async def status_page(req: Request):
    if not browser_ok(req):
        return RedirectResponse("/login", status_code=303)
    d, rows = status_data(), []
    for p in d["peers"]:
        models = " ".join(f"<span class={'on' if s == 'loaded' else 'off'}>{escape(m)}</span>"
                          for m, s in sorted(p["models"].items()))
        gpus = "".join(
            f"<tr><td>{escape(str(g.get('index')))}</td><td>{escape(str(g.get('name', '')))}</td>"
            f"<td>{escape(str(g.get('util', '')))}%</td>"
            f"<td>{escape(str(g.get('mem_used', '')))} / {escape(str(g.get('mem_total', '')))} MiB</td>"
            f"<td>{escape(str(g.get('temp', '')))} C</td><td>{escape(str(g.get('power', '')))} W</td></tr>"
            for g in p["gpus"])
        rows.append(f"<section><h2>{escape(p['host'])} <small>{'up' if p['ok'] else 'UNREACHABLE'}, "
                    f"heartbeat {p['age_s']}s ago, {p['inflight']} in flight</small></h2>"
                    f"<p>{models or 'no models'}</p><table><tr><th>gpu<th>name<th>util<th>vram<th>temp<th>power"
                    f"{gpus}</table></section>")
    return HTMLResponse(f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=2><title>llm status</title><style>body{{font:15px system-ui;background:#111;
color:#ddd;margin:0;padding:1.2rem}}h1{{font-size:1.2rem}}h2{{font-size:1rem;margin:1.4rem 0 .3rem}}
small{{color:#888;font-weight:400}}table{{border-collapse:collapse;width:100%;max-width:46rem}}
td,th{{padding:.25rem .6rem;border-bottom:1px solid #262626;text-align:left}}
.on{{background:#14532d;padding:.1rem .4rem;border-radius:.3rem;margin-right:.3rem}}
.off{{background:#333;padding:.1rem .4rem;border-radius:.3rem;margin-right:.3rem;color:#999}}
a{{color:#6cf}}</style><h1>local models <small>{d['ts']} - <a href=/>chat</a> - <a href=/status.json>json</a></small></h1>
{''.join(rows) or '<p>no peers registered</p>'}""")


async def health(_req: Request):
    return JSONResponse({"ok": True, "peers": len(peers)})


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(poller())
    yield
    task.cancel()


app = Starlette(lifespan=lifespan, routes=[
    Route("/health", health), Route("/peers", list_peers),
    Route("/login", login, methods=["GET", "POST"]),
    Route("/status", status_page), Route("/status.json", status_json),
    Route("/peers/register", register, methods=["POST"]), Route("/peers/deregister", register, methods=["POST"]),
    Route("/v1/models", models), Route("/models", models),
    Route("/{path:path}", proxy, methods=["GET", "POST"]),
])
