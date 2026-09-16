"""Stdio MCP server with two tools for llama-server: `search` (web search) and `fetch` (read a URL).

llama-server spawns this per tool call (tools/server/README.md "MCP servers"): one JSON-RPC message
per line on stdin/stdout. It is deliberately dependency-light (httpx, plus `ddgs` for keyless search).

Security (this runs on cluster nodes behind an internet-facing endpoint):
  * fetch refuses non-http(s) schemes, URLs with credentials, and any host that resolves to a
    non-public address (private, loopback, link-local, CGNAT, multicast, reserved, IPv6 ULA ...).
    The check runs after DNS resolution, the connection is pinned to the vetted IP (no DNS
    rebinding), and it is repeated on every redirect hop.
  * responses are capped in size and time; only textual content is returned.
  * search uses DuckDuckGo (no key), or Brave Search when ~/.config/local-model-serve/brave-api-key
    (mode 0600) exists.
"""
import html
import ipaddress
import json
import os
import re
import socket
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = int(os.environ.get("WEB_FETCH_MAX_BYTES", 2 * 1024 * 1024))
MAX_CHARS = int(os.environ.get("WEB_FETCH_MAX_CHARS", 20000))
TIMEOUT = float(os.environ.get("WEB_FETCH_TIMEOUT", 15))
MAX_REDIRECTS = 5
UA = "Mozilla/5.0 (compatible; llm-garylvov-web/1.0; +https://llm.garylvov.com)"
CONF = Path(os.environ.get("LLM_CONFIG_DIR", Path.home() / ".config/local-model-serve"))
# Beyond ipaddress's is_global, name the ranges the operator asked for explicitly.
BLOCKED = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12",
    "192.0.0.0/24", "192.168.0.0/16", "198.18.0.0/15", "224.0.0.0/4", "240.0.0.0/4",
    "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8")]


class Refused(Exception):
    pass


def log(msg: str) -> None:
    # stderr only (stdout is the protocol); never log page contents or queries beyond their length
    print(f"[web-mcp] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- SSRF guard
def ip_allowed(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_global and not any(a in n for n in BLOCKED if n.version == a.version)


def resolve(host: str) -> list[str]:
    try:
        ips = sorted({ai[4][0] for ai in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})
    except socket.gaierror:
        ips = []
    if not ips:  # Oscar's resolver misses some public names: ask Cloudflare DoH by IP
        try:
            r = httpx.get("https://1.1.1.1/dns-query", params={"name": host, "type": "A"},
                          headers={"accept": "application/dns-json"}, timeout=5)
            ips = [a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1]
        except (httpx.HTTPError, ValueError):
            ips = []
    return ips


def vet(url: str) -> tuple[httpx.URL, str]:
    """Return (url, pinned ip) or raise Refused."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise Refused(f"scheme '{parts.scheme}' is not allowed (http/https only)")
    if parts.username or parts.password:
        raise Refused("URLs with credentials are not allowed")
    host = parts.hostname
    if not host:
        raise Refused("URL has no host")
    try:  # literal IP
        ips = [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        ips = resolve(host)
    if not ips:
        raise Refused(f"could not resolve {host}")
    bad = [ip for ip in ips if not ip_allowed(ip)]
    if bad:
        raise Refused(f"{host} resolves to a non-public address; internal and private networks are off limits")
    return httpx.URL(url), ips[0]


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)

    def text(self) -> str:
        t = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.out))
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def fetch(url: str, max_chars: int = MAX_CHARS) -> dict:
    t0 = time.monotonic()
    max_chars = max(500, min(int(max_chars or MAX_CHARS), MAX_CHARS))
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT), follow_redirects=False, trust_env=False) as c:
        for hop in range(MAX_REDIRECTS + 1):
            u, ip = vet(url)
            pinned = u.copy_with(host=ip)
            headers = {"Host": u.netloc.decode(), "User-Agent": UA,
                       "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"}
            ext = {"sni_hostname": u.host} if u.scheme == "https" else {}
            with c.stream("GET", pinned, headers=headers, extensions=ext) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    url = urljoin(str(u), r.headers["location"])
                    continue
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not (ctype.startswith("text/") or ctype in ("application/json", "application/xml",
                                                                          "application/xhtml+xml")):
                    raise Refused(f"content type '{ctype}' is not text")
                body = bytearray()
                for chunk in r.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES or time.monotonic() - t0 > TIMEOUT:
                        break
                truncated_bytes = len(body) > MAX_BYTES
                raw = bytes(body[:MAX_BYTES]).decode(r.encoding or "utf-8", "replace")
                status = r.status_code
            break
        else:
            raise Refused(f"more than {MAX_REDIRECTS} redirects")
    title = ""
    if "html" in ctype or raw.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        p = TextExtractor()
        p.feed(raw)
        text, title = p.text(), html.unescape(p.title.strip())
    else:
        text = raw
    return {"url": str(u), "status": status, "title": title, "content_type": ctype,
            "truncated": truncated_bytes or len(text) > max_chars, "text": text[:max_chars],
            "elapsed_s": round(time.monotonic() - t0, 2)}


# --------------------------------------------------------------------------- search
# Tried in order; "auto" is ddgs's own rotation and makes a decent last resort.
SEARCH_BACKENDS = ("brave", "bing", "yahoo", "duckduckgo", "mojeek", "auto")


def search(query: str, max_results: int = 6) -> dict:
    t0 = time.monotonic()
    n = max(1, min(int(max_results or 6), 10))
    key_file = CONF / "brave-api-key"
    if key_file.exists() and (key_file.stat().st_mode & 0o077) == 0:
        r = httpx.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                      headers={"X-Subscription-Token": key_file.read_text().strip(), "Accept": "application/json"},
                      timeout=TIMEOUT)
        r.raise_for_status()
        results = [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": re.sub("<[^>]+>", "", x.get("description", ""))}
                   for x in r.json().get("web", {}).get("results", [])[:n]]
        engine = "brave"
    else:
        # Keyless search: any single engine rate-limits us into "No results found" (measured
        # 2026-09-16 — duckduckgo, google and mojeek all returned nothing while brave, bing and
        # yahoo answered the same query). Rotate; fail only when every engine has.
        from ddgs import DDGS
        errors, results, engine = [], [], None
        for backend in SEARCH_BACKENDS:
            try:
                hits = DDGS(timeout=int(TIMEOUT)).text(query, max_results=n, backend=backend)
                results = [{"title": x.get("title", ""), "url": x.get("href", ""), "snippet": x.get("body", "")}
                           for x in hits]
                if results:
                    engine = backend
                    break
            except Exception as e:          # per-engine failures are routine; try the next one
                errors.append(f"{backend}: {type(e).__name__}")
        if not results:
            raise RuntimeError("every search engine failed or was empty (" + "; ".join(errors) + ")")
    return {"query": query, "engine": engine, "results": results, "elapsed_s": round(time.monotonic() - t0, 2)}


# --------------------------------------------------------------------------- MCP stdio JSON-RPC
TOOLS = [
    {"name": "search",
     "description": "Search the web. Returns titles, URLs and snippets. Use `web_fetch` on a result URL to read it.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "search query"},
         "max_results": {"type": "integer", "description": "1-10, default 6"}}, "required": ["query"]}},
    {"name": "fetch",
     "description": "Fetch a public http(s) web page and return its readable text (truncated). "
                    "Internal/private network addresses are refused.",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string", "description": "absolute http(s) URL"},
         "max_chars": {"type": "integer", "description": f"max characters of text, default {MAX_CHARS}"}},
         "required": ["url"]}},
]


def call_tool(name: str, args: dict) -> dict:
    try:
        if name == "search":
            out = search(str(args.get("query", "")), args.get("max_results", 6))
            log(f"search ok ({len(out['results'])} results, {out['elapsed_s']}s)")
        elif name == "fetch":
            out = fetch(str(args.get("url", "")), args.get("max_chars", MAX_CHARS))
            log(f"fetch ok status={out['status']} chars={len(out['text'])} {out['elapsed_s']}s")
        else:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False}
    except Refused as e:
        log(f"{name} refused: {e}")
        return {"content": [{"type": "text", "text": f"refused: {e}"}], "isError": True}
    except Exception as e:  # noqa: BLE001 - report, never crash the stdio loop
        log(f"{name} failed: {type(e).__name__}")
        return {"content": [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}], "isError": True}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # notification (e.g. notifications/initialized)
            continue
        if method == "initialize":
            result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "web", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            prm = msg.get("params", {})
            result = call_tool(prm.get("name", ""), prm.get("arguments") or {})
        elif method == "ping":
            result = {}
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid,
                                         "error": {"code": -32601, "message": f"method not found: {method}"}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
