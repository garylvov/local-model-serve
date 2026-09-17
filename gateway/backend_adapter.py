#!/usr/bin/env python3
"""Adapter that lets a non-llama.cpp engine (vLLM, DwarfStar, ...) register with the llm gateway
as an ordinary peer. It launches the engine as a subprocess, exposes a /models endpoint in the
schema the gateway's poll() already understands ({"data":[{"id", "status":{"value":...}}]}), and
proxies every other path straight through to the engine's own port. It also heartbeats itself to
the gateway's POST /peers/register, the same call bin/llm makes for the llama.cpp router - the
gateway does not need to know the difference (design point 3: "quick-tunnel peer test proved this
path").

Usage: backend_adapter.py --cmd '<shell command>' --cwd DIR --engine-port N --listen-port N
                           --model-id NAME --health-path /health --models-path /v1/models
                           --gateway-url URL --key-file PATH [--env K=V ...]
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

engine_proc = None
args = None
client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None))


async def engine_up() -> bool:
    try:
        r = await client.get(f"http://127.0.0.1:{args.engine_port}{args.health_path}", timeout=3)
        return r.status_code < 500
    except httpx.HTTPError:
        return False


async def models(_req: Request):
    up = await engine_up()
    status = "loaded" if up else "loading"
    return JSONResponse({"data": [{"id": args.model_id, "status": {"value": status}}]})


async def health(_req: Request):
    return JSONResponse({"status": "ok"} if await engine_up() else {"status": "starting"}, status_code=200 if await engine_up() else 503)


async def proxy(req: Request):
    url = f"http://127.0.0.1:{args.engine_port}{req.url.path}"
    if req.url.query:
        url += f"?{req.url.query}"
    body = await req.body()
    headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
    rq = client.build_request(req.method, url, content=body, headers=headers)
    resp = await client.send(rq, stream=True)

    async def gen():
        async for chunk in resp.aiter_raw():
            yield chunk
        await resp.aclose()

    return StreamingResponse(gen(), status_code=resp.status_code,
                              headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")})


async def heartbeat_loop():
    key = open(args.key_file).read().strip()
    my_url = f"http://{os.uname().nodename}:{args.listen_port}"
    while True:
        try:
            await client.post(f"{args.gateway_url}/peers/register", json={"url": my_url, "host": os.uname().nodename},
                              headers={"Authorization": f"Bearer {key}"}, timeout=8)
        except httpx.HTTPError as e:
            print(f"heartbeat failed: {e}", file=sys.stderr, flush=True)
        await asyncio.sleep(15)


async def lifespan(_app):
    global engine_proc
    env = os.environ.copy()
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    print(f"backend_adapter: launching: {args.cmd}", flush=True)
    engine_proc = subprocess.Popen(args.cmd, shell=True, cwd=args.cwd, env=env)
    hb = asyncio.create_task(heartbeat_loop())
    yield
    hb.cancel()
    if engine_proc and engine_proc.poll() is None:
        engine_proc.send_signal(signal.SIGTERM)
        try:
            engine_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            engine_proc.kill()


def build_app():
    return Starlette(routes=[
        Route("/models", models), Route("/v1/models", models), Route("/health", health),
        Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ], lifespan=lifespan)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", required=True)
    p.add_argument("--cwd", required=True)
    p.add_argument("--engine-port", type=int, required=True)
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--health-path", default="/health")
    p.add_argument("--models-path", default="/v1/models")
    p.add_argument("--gateway-url", required=True)
    p.add_argument("--key-file", required=True)
    p.add_argument("--env", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(build_app(), host="0.0.0.0", port=args.listen_port, log_level="info")
