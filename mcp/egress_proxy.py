"""Egress proxy for the browser tool: forwards only to public internet addresses.

The Playwright MCP browser runs on a GPU node inside a private network, and a page it visits can
point it anywhere. Every browser request goes through this proxy (HTTP CONNECT for https, plain
forwarding for http). The proxy resolves the target itself, refuses anything that is not a public
address (same rules as web_server.ip_allowed), and connects to the vetted IP so a hostile DNS
answer cannot re-point the connection after the check.

    python mcp/egress_proxy.py <port>        # listens on 127.0.0.1 only
"""
import asyncio
import sys
from urllib.parse import urlsplit

from web_server import ip_allowed, resolve

BUF = 1 << 16


def log(msg: str) -> None:
    print(f"[egress] {msg}", file=sys.stderr, flush=True)


def vetted_ip(host: str) -> str | None:
    ips = resolve(host.strip("[]"))
    # every answer must be public: a mixed answer is how DNS-rebinding attacks start
    return ips[0] if ips and all(ip_allowed(ip) for ip in ips) else None


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(BUF):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def refuse(w: asyncio.StreamWriter, host: str) -> None:
    log(f"refused {host}: not a public address")
    w.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    await w.drain()
    w.close()


async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 15)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
        cw.close()
        return
    method, target, _ = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ", 2)
    if method == "CONNECT":                                   # https: host:port, then raw bytes
        host, _, port = target.rpartition(":")
        ip = await asyncio.to_thread(vetted_ip, host)
        if not ip:
            return await refuse(cw, host)
        try:
            ur, uw = await asyncio.wait_for(asyncio.open_connection(ip, int(port)), 15)
        except (OSError, asyncio.TimeoutError, ValueError):
            cw.close()
            return
        cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await cw.drain()
    else:                                                     # plain http: absolute-form URL
        u = urlsplit(target)
        if u.scheme != "http" or not u.hostname:
            return await refuse(cw, target)
        ip = await asyncio.to_thread(vetted_ip, u.hostname)
        if not ip:
            return await refuse(cw, u.hostname)
        try:
            ur, uw = await asyncio.wait_for(asyncio.open_connection(ip, u.port or 80), 15)
        except (OSError, asyncio.TimeoutError):
            cw.close()
            return
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        uw.write(head.replace(target.encode(), path.encode(), 1))
        await uw.drain()
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main(port: int) -> None:
    server = await asyncio.start_server(handle, "127.0.0.1", port)
    log(f"listening on 127.0.0.1:{port}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
