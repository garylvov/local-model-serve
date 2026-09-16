#!/usr/bin/env bash
# Browser tool (Playwright MCP, stdio) for llama-server, fenced in by mcp/egress_proxy.py:
# every request the browser makes goes through the proxy, which only reaches public addresses.
# Chrome skips proxies for loopback unless told otherwise, hence --proxy-bypass-list=<-loopback>.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$ROOT"

port="$(mcp/.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')"
mcp/.venv/bin/python mcp/egress_proxy.py "$port" 2>>"${LLM_BROWSER_LOG:-/dev/null}" &
proxy=$!
trap 'kill $proxy 2>/dev/null' EXIT
for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break; sleep 0.1; done

chrome="${LLM_BROWSER_EXECUTABLE:-$(ls ~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell 2>/dev/null | tail -1)}"
cfg="$(mktemp)"; trap 'kill $proxy 2>/dev/null; rm -f "$cfg"' EXIT
cat > "$cfg" <<EOF
{"browser": {"launchOptions": {"headless": true, "executablePath": "$chrome",
  "args": ["--no-sandbox", "--proxy-server=http://127.0.0.1:$port", "--proxy-bypass-list=<-loopback>"]},
  "contextOptions": {"ignoreHTTPSErrors": false}, "isolated": true}}
EOF

export npm_config_cache="$ROOT/mcp/.npm"
# Run from an EMPTY workspace: Playwright MCP confines file access (browser_file_upload, file://) to
# the workspace roots, so the model can never attach repo files or ~/.config keys to a web form.
ws="$(mktemp -d)"; trap 'kill $proxy 2>/dev/null; rm -rf "$cfg" "$ws"' EXIT
cd "$ws"
npx -y @playwright/mcp@latest --config "$cfg" --isolated --headless --image-responses omit
