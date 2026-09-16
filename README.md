# local-model-serve

Serve local LLMs with [llama.cpp](https://github.com/ggml-org/llama.cpp) from one or many GPU
machines behind a single OpenAI- and Anthropic-compatible endpoint, with a browser chat, live
hardware view, and web tools.

```
clients ──> gateway (:4000) ──> llama-server router (:8080) on each machine ──> one process per model
```

Machines register themselves with the gateway; clients only ever talk to the gateway.

## Quick start

```bash
git clone https://github.com/garylvov/local-model-serve && cd local-model-serve
cp config.example.env config.env      # optional: public URL, gateway host, Cloudflare zone
bin/llm join                          # build llama.cpp, start the router, load the preset's models
bin/llm ls                            # what is loaded, on which GPUs, how fast
```

The API key is written to `~/.config/local-model-serve/auth.env`. Point any OpenAI or Anthropic
client at the gateway with it:

```bash
export OPENAI_BASE_URL=http://<gateway>:4000/v1        ANTHROPIC_BASE_URL=http://<gateway>:4000
export OPENAI_API_KEY=$LLM_API_KEY                     ANTHROPIC_AUTH_TOKEN=$LLM_API_KEY
```

## Setups

<details>
<summary><b>1. Single machine / workstation (no scheduler)</b></summary>

Everything runs on one box. Leave `LLM_GATEWAY_HOST` empty.

```bash
bin/llm join              # router on :8080
bin/llm passwd            # browser password for the chat UI
bin/llm gateway up        # gateway on :4000 (chat, /status, /models, the API)
```

Open `http://localhost:4000`. To add more machines later, run `bin/llm join` on each with
`LLM_GATEWAY_HOST` (same network) or `LLM_PUBLIC_URL` (anywhere) set in `config.env`.

</details>

<details>
<summary><b>2. Slurm cluster, gateway on a login node</b></summary>

The gateway needs no GPU, so run it on a login node where it outlives jobs. GPU jobs come and
go and register themselves.

```bash
# config.env (shared by the login node and GPU nodes)
LLM_GATEWAY_HOST=login-node-hostname
LLM_PUBLIC_URL=https://llm.example.com      # optional, if exposed through a tunnel
```

```bash
# on the login node, once
bin/llm passwd && bin/llm gateway up        # runs in tmux session `llm-gateway`

# GPU nodes: submit a job ...
sbatch slurm/serve.sbatch
# ... or join an existing allocation
ssh <gpu-node> 'cd /path/to/local-model-serve && bin/llm join'
```

A node that stops heartbeating (job ended, node lost) drops out of routing after 90 s.

</details>

<details>
<summary><b>3. Remote machines (home rig, cloud box)</b></summary>

A machine that cannot reach the gateway's private host joins over the public URL. `llm join`
opens a Cloudflare quick tunnel (no account needed) so the gateway can reach it back.

```bash
# config.env on that machine
LLM_PUBLIC_URL=https://llm.example.com
```

Install [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/),
copy `auth.env` from the gateway machine (`bin/llm auth print-client | ssh rig 'cat > ~/.config/local-model-serve/auth.env'`),
then `bin/llm join`.

</details>

<details>
<summary><b>Exposing the gateway publicly (Cloudflare Tunnel)</b></summary>

1. Create a tunnel: [Cloudflare Tunnel setup guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/).
2. Run its connector on the gateway machine (`cloudflared tunnel run --token-file ...`).
3. Add a public hostname pointing at `http://<gateway-host>:4000`, either in the Cloudflare
   dashboard or with `scripts/cf-route.sh` (dry-run by default; `--apply` to write):

   ```bash
   CF_ZONE=example.com LLM_PUBLIC_URL=https://llm.example.com \
     scripts/cf-route.sh --service http://gateway-host:4000
   ```

The gateway redirects HTTP to HTTPS and sends HSTS when it sees Cloudflare headers. The browser
UI needs a password (`bin/llm passwd`); the API needs the key.

</details>

## Configuration

<details>
<summary><b>config.env</b></summary>

See [`config.example.env`](config.example.env). All settings are optional; environment variables
override the file.

| Variable | Purpose |
| --- | --- |
| `LLM_GATEWAY_HOST` | Gateway hostname on the private network |
| `LLM_PUBLIC_URL` | Gateway public URL, for remote machines and clients |
| `CF_ZONE`, `CF_API_TOKEN_FILE` | For `scripts/cf-route.sh` / `scripts/cf-machine.sh` |
| `LLM_PRESET` | Force a preset instead of choosing by GPU count |
| `LLM_MODELS_DIR` | Where weights go (default `models/`) |

</details>

<details>
<summary><b>Presets: which model runs on which GPUs</b></summary>

`presets/*.ini` are llama.cpp [model presets](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#model-presets).
`bin/llm` picks one by GPU count (`1 → single-24g`, `2 → dual-24g`, `4 → quad-24g`,
`8 → 8x-24g`). One section per model:

```ini
[qwen3.8-27b@1]
load-on-startup = true
hf-repo = unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
device = CUDA5
ctx-size = 65536
spec-type = draft-mtp
spec-draft-hf = unsloth/Qwen3.8-27B-GGUF
spec-draft-model = MTP/mtp-Qwen3.8-27B-Q4_0.gguf
```

Sections named `name@1`, `name@2`, … are replicas: clients ask for `name` and the gateway sends
each request to the least busy copy. Placement can also be edited from the **Models** page.

</details>

<details>
<summary><b>Tools: web search, fetch, browser</b></summary>

Registered in [`mcp/web.json`](mcp/web.json) and run by llama-server (no API keys):

- `web_search` rotates over several engines; `web_fetch` reads a page as text.
- `browser` is [Playwright MCP](https://github.com/microsoft/playwright-mcp) in headless Chrome.

Fetch and the browser only reach public addresses: private, loopback and link-local targets are
refused after DNS resolution, and the browser runs through an egress proxy
(`mcp/egress_proxy.py`) with no access to local files. llama.cpp's filesystem/shell tools are
not enabled.

</details>

## Commands

<details>
<summary><b>bin/llm</b></summary>

| Command | |
| --- | --- |
| `join` / `leave` | attach / detach this machine |
| `serve` / `stop` | start / stop the router |
| `up <model>` / `down <model>` | load / unload (fuzzy names) |
| `pause` / `resume` | unload everything / reload it |
| `ls` | loaded models, GPUs, speed |
| `pull <model>` | download only |
| `gateway up\|down\|status` | run the gateway |
| `passwd` | browser password |
| `build <variant>` / `builds` | llama.cpp builds from [`catalog/builds.yaml`](catalog/builds.yaml) |
| `tunnel up\|down` | this machine's Cloudflare tunnel |

</details>

<details>
<summary><b>Loading very large models faster</b></summary>

llama.cpp reads weights on a single thread, which is slow on network filesystems, and it never
releases an unloaded model's pages from the page cache. `bin/llm` handles both:

- `llm up <model>` reads a large model's files in parallel first (over `LLM_PREWARM_MIN_GIB`,
  default 32), so the loader finds them in RAM. `LLM_NO_PREWARM=1` skips this.
- `llm down <model>` evicts its files from the page cache, so the next model has room.
  `LLM_KEEP_CACHE=1` keeps them for a faster reload.

Manually: `python3 scripts/weights_cache.py warm|evict <hf-repo:quant>`.
Measured for an 802 GiB model over NFS: 47+ min cold, 4 min warm-up + 1 min load.

</details>
