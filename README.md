# local-model-serve

Run local LLMs behind one OpenAI- and Anthropic-compatible endpoint, with a browser chat, a live
hardware view and web tools. One command on one machine; the same repo scales to several machines,
a Slurm cluster and a public URL when you want it.

```
clients ──> gateway (:4000) ──> engine on each machine ──> one process per model
                                llama.cpp (default), vLLM, DwarfStar
```

## Quick start

```bash
git clone https://github.com/garylvov/local-model-serve && cd local-model-serve
bin/llm local          # build llama.cpp, start the router + gateway on 127.0.0.1, print the URL
```

Open `http://127.0.0.1:4000`: chat, a live **Hardware** page and a **Models** page (load, unload,
GPU placement). Nothing is exposed off the machine; a browser password is generated on first run.

Point any OpenAI or Anthropic client at the same address, with the key from
`~/.config/local-model-serve/auth.env`:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:4000/v1        ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export OPENAI_API_KEY=$LLM_API_KEY                     ANTHROPIC_AUTH_TOKEN=$LLM_API_KEY
```

Everything below is optional: more machines, a scheduler, a public URL.

## What it downloads, and where

`bin/llm local` writes everything inside the repo directory, plus a small config directory. Nothing
is sent anywhere: weights come from Hugging Face, and requests stay on the machine.

| Path | What | Size |
| --- | --- | --- |
| `vendor/llama.cpp/` | llama.cpp source + CUDA build (first run only, a few minutes) | ~2 GB |
| `models/` | model weights (GGUF), in llama.cpp's own cache layout (`LLAMA_CACHE`) | as big as the models |
| `run/<host>/` | logs, PIDs, the preset in use | small |
| `~/.config/local-model-serve/` | `api-key`, `auth.env` (client key + URL), `passwd` (scrypt hash) | tiny, mode 0600 |

**Which weights.** `bin/llm` picks a preset from the GPU count (`presets/single-24g.ini` for one
GPU, `dual-24g`, `quad-24g`, `8x-24g`), and each section names a Hugging Face repo and quant, for
example `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` (16.4 GiB) on a single 24 GB GPU. Only models marked
`load-on-startup` are fetched at first run; the rest download on `llm up <model>` or from the
**Models** page (which shows the size and asks first). `llm pull <model>` downloads without loading.

**Moving the weights:** set `LLM_MODELS_DIR=/somewhere/big` (or `models/` can be a symlink). Disk
is the main thing to plan: a 24 GB-GPU model is ~16 GiB, a 180 GiB-VRAM model is ~127 GiB, and the
big MoE experiments here have run to 800 GiB.

**Cleaning up:** `llm down <model>` also drops that model's pages from the OS page cache;
weights stay on disk until you delete them from `models/`.

## Setups

<details>
<summary><b>1. One machine (the default) — what <code>llm local</code> sets up</b></summary>

`bin/llm local` is the whole setup: it builds llama.cpp if needed, starts the router, starts the
gateway bound to `127.0.0.1`, generates a browser password if there isn't one, and prints the URLs.
No config file, no heartbeat, no tunnel.

```bash
bin/llm local             # everything
bin/llm ls                # what is loaded, on which GPUs, how fast
bin/llm up <model>        # load another model from the preset (fuzzy name)
bin/llm stop && bin/llm gateway down
```

The preset is chosen from the GPU count (`presets/single-24g.ini` for one GPU, and so on), so the
models it offers match the machine. To add other machines later, see the next two sections.

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

## Backends

<details>
<summary><b>Serving a model with something other than llama.cpp (vLLM, DwarfStar, ...)</b></summary>

Every model is served by a **backend**: an engine process that answers OpenAI-compatible requests.
[`catalog/backends.yaml`](catalog/backends.yaml) lists the ones this repo knows about. A preset
section names one with `backend = <name>`; leaving it out means `backend = llamacpp`, so every
preset written before this feature existed is unchanged.

```ini
[qwen-small-vllm]
backend = vllm
hf-repo = Qwen/Qwen2.5-0.5B-Instruct
device = CUDA7
ctx-size = 4096
parallel = 1
```

- **`backend = llamacpp`** (default): served by the shared llama-server router exactly as before
  (`presets/*.ini`, `llm serve`/`up`/`down`). One process, many models, on-demand load/unload.
- **any other backend**: `llm up <model>` starts that engine as its **own process on its own
  port**, in front of a small adapter (`gateway/backend_adapter.py`) that translates its `/models`
  (or `/v1/models`) response into the shape the gateway already polls
  (`{"data":[{"id", "status":{"value":"loaded"}}]}`) and proxies every other request straight
  through. The adapter heartbeats itself to the gateway's `POST /peers/register` — the same call
  `bin/llm` makes for the router — so the gateway treats it as an ordinary peer and routes to it by
  model name. `llm ls` shows a BACKEND column and merges these processes into the same table;
  `llm down <model>` stops the engine and deregisters it.

**Adding a new engine** needs no code if it's already OpenAI-compatible: add an entry to
`catalog/backends.yaml` with a `cmd:` template (llama-swap's contract - opaque command string +
`checkEndpoint`) and a `port_base`. Template variables: `port`, `model_path`, `repo`, `quant`,
`gpu_ids`, `gpu_count`, `ctx_size`, `parallel`, `models_dir`, `extra_args`, `mmproj_path`,
`spec_model`, `spec_json`. A few typed fields are mapped per engine rather than left as raw flags,
because they need translation, not just formatting:

| Typed field | llama.cpp | vLLM | DwarfStar |
| --- | --- | --- | --- |
| GPU placement | `--device CUDA0,CUDA1,...` | `CUDA_VISIBLE_DEVICES` + `--tensor-parallel-size` | `CUDA_VISIBLE_DEVICES` |
| context size | `--ctx-size` | `--max-model-len` | `--ctx-size` |
| parallel/concurrency | `--parallel` | implicit (continuous batching) | not yet measured |
| quant selection | `-hf repo:quant` (GGUF) | repo id only (safetensors bf16/fp8/awq/gptq; GGUF unsupported for most archs) | its own GGUF variant, not llama.cpp's |
| draft/speculative model | `spec-*` flags (`draft-mtp`/`draft`) | `--speculative-config '<json>'` | not documented |
| multimodal projector | `-hf` auto-picks mmproj, or `--mmproj-url` | none - vision tower loads from the model repo | unsupported (text-only) |
| prefix caching | text only - llama.cpp logs `cache_reuse is not supported by multimodal` for images | automatic (V1 default); text measured working | unknown |

Anything else goes through `extra_args = --foo bar` untouched. `scripts/render_backend_cmd.py`
**rejects an unsupported combination at launch** (e.g. `spec-draft-model` set on an engine whose
`supports.speculative` is missing/`unknown`, or an mmproj on a text-only engine) with the exact
section and flag named, instead of silently dropping it.

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
