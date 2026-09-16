"""Drive the same loop the WebUI runs: GET /tools, chat with tools, execute tool_calls via POST /tools."""
import json, os, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "https://llm.garylvov.com"
KEY = open(os.path.expanduser("~/.config/local-model-serve/api-key")).read().strip()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "llm-agent-loop-test/1.0"}
def req(method, path, body=None):
    r = urllib.request.Request(BASE + path, method=method, headers=H, data=json.dumps(body).encode() if body is not None else None)
    t = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        return json.loads(resp.read()), time.time() - t
tools, _ = req("GET", "/tools")
defs = [t["definition"] for t in tools]
msgs = [{"role": "system", "content": "You have web tools. For questions about current facts, call web_search, then web_fetch the most relevant result, then answer briefly and cite the URL you fetched."},
        {"role": "user", "content": "What is the newest stable Python 3 release right now, and when was it released?"}]
for turn in range(6):
    out, dt = req("POST", "/v1/chat/completions", {"model": "qwen3.8-27b", "messages": msgs, "tools": defs, "max_tokens": 1500, "temperature": 0.2})
    m = out["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    print(f"--- turn {turn}: model {dt:.1f}s, finish={out['choices'][0]['finish_reason']}, tool_calls={len(calls)}")
    msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "reasoning_content")})
    if not calls:
        print("ANSWER:", (m.get("content") or "").strip())
        break
    for c in calls:
        args = json.loads(c["function"]["arguments"] or "{}")
        res, tdt = req("POST", "/tools", {"tool": c["function"]["name"], "params": args})
        text = json.dumps(res)
        print(f"    call {c['function']['name']}({json.dumps(args)}) -> {tdt:.2f}s, {len(text)} chars: {text[:160]}")
        msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": text[:24000]})
