# local-model-serve

Serve local LLMs for agentic coding with llama.cpp, so that agents can call
`https://llm.garylvov.com` like any remote API and never have to know which machine
answered. The design is deliberately thin: llama.cpp's own **router mode** does the
model management, and everything else here is glue.

```
agent ──https──> Cloudflare ──> gateway (this repo, :4000) ──> llama-server ROUTER (:8080, one per machine)
                               merges peers, routes by model            └── child llama-server per loaded model
                               browser chat + /status
```

* **`bin/llm`** (≈300 lines of bash; `bin/lms` is a symlink) wraps the router and curl.
* **`presets/*.ini`** are llama.cpp model presets: one section per model, carrying its
  HF repo/quant, GPUs (`device = CUDA6,CUDA7`), context, and flags. `llm serve` picks the
  preset by GPU count (`8 → oscar-8x`, `4 → quad-24g`, `2 → dual-24g`, `1 → single-24g`),
  or `LLM_PRESET=<name>`.
* **`catalog/models.yaml`** is documentation: what *can* run, with verified HF sizes,
  architecture support, per-GPU-count fit, and known upstream bugs. What *is* running comes
  from the router's `GET /models`.
* **`catalog/builds.yaml` + `scripts/build-llama.sh <variant>`** build llama.cpp variants
  (`master` pinned, plus `poolside-laguna` and `glm53` for models that need a fork/PR).
* **`gateway/llm_gateway.py`** (one file) is the only public entry point: peers heartbeat to
  it, it merges their model lists, routes each request to a peer that has that model loaded,
  streams responses through unchanged, and serves the browser chat + `/status`.

Weights live in `models/` (gitignored, `LLAMA_CACHE`), builds in `vendor/` and runtime state
in `run/` — none of it is committed.

## Quick start

### Attach a new GPU node (one command)

```bash
cd /oscar/data/stellex/glvov/local-model-serve && bin/llm join
```

That is the whole thing, and it is safe to re-run. `llm join` detects the GPUs, builds llama.cpp
only if the pinned build is missing, picks the preset for the GPU count, starts the router in tmux
`llm` (which auto-loads the preset's `load-on-startup` model), starts the 30 s heartbeat to
`http://$LLM_GATEWAY_HOST:4000` (default `login009`), and prints the model table. `llm leave`
deregisters, stops the heartbeat, the tunnel (if any) and the router.

On Oscar, from a login node:

```bash
salloc -p gpu --gres=gpu:8 -c 60 --mem=900G -t 24:00:00     # new allocation, lands you on the node
cd /oscar/data/stellex/glvov/local-model-serve && bin/llm join

# or join a node inside an allocation you already hold:
ssh gpu2260 'cd /oscar/data/stellex/glvov/local-model-serve && bin/llm join'          # preferred: tmux outlives the step
srun --overlap --jobid <jobid> --pty bash                                             # alternative interactive shell
```

Prefer the `ssh <node>` form (Slurm allows it on nodes where you hold an allocation): tmux sessions
started inside an `srun` step can be cleaned up when that step exits, while the ssh form leaves the
router running for the life of the job. When the job ends the node simply stops heartbeating and the
gateway drops it within 90 s (it polls every 10 s), so nothing has to be cleaned up by hand.

### New Oscar node (or any Linux + NVIDIA machine), step by step

```bash
git clone <repo> /oscar/data/stellex/glvov/local-model-serve && cd $_
module load cuda/12.9.0-cinr          # only where modules exist
scripts/build-llama.sh                # CUDA build, arch auto-detected
bin/llm auth init                     # generates ~/.config/local-model-serve/api-key + auth.env (0600)
bin/llm pull qwen                     # llama.cpp's own downloader into models/ (resumable)
bin/llm serve                         # router on :8080 in tmux 'llm'
bin/llm up qwen                       # load a model (fuzzy name match)
bin/llm ls                            # what is loaded, on which GPUs, decode tok/s
```

To join the shared gateway, add to `~/.config/local-model-serve/auth.env`:

```dotenv
LLM_GATEWAY_URL=http://gpu2260:4000       # on-cluster peers talk directly
LLM_PEER_URL=http://<this-host>:8080
```

`llm serve` then heartbeats every 30 s (tmux `llm-heartbeat`); `llm pause` deregisters.

### Home rig (any machine on the open internet)

```bash
# 1. cloudflared (no Cloudflare account needed for a quick tunnel)
mkdir -p ~/.local/bin && curl -fsSL -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared
# 2. the shared key: copy auth.env from an Oscar machine (never paste it into a terminal log)
ssh oscar 'cd /oscar/data/stellex/glvov/local-model-serve && bin/llm auth print-client' \
  | (umask 077; mkdir -p ~/.config/local-model-serve && cat > ~/.config/local-model-serve/auth.env)
# 3. one command
git clone <repo> ~/local-model-serve && cd ~/local-model-serve && bin/llm join
```

What `llm join` does off-cluster, with no arguments:

* **Peer → gateway:** `login009` does not resolve outside Oscar, so `llm` detects that and sends
  heartbeats to `https://llm.garylvov.com` (on Oscar it uses `http://login009:4000`). The
  heartbeat authenticates with the bearer key from `auth.env`.
* **Gateway → peer:** a home rig has no public address, so `llm join` starts a cloudflared
  **quick tunnel** (`cloudflared tunnel --url http://127.0.0.1:8080`, tmux `llm-tunnel`): no
  account, no DNS, no API writes. It parses the `https://<random>.trycloudflare.com` URL and the
  heartbeat registers that. The URL changes whenever cloudflared restarts; every heartbeat
  re-reads the newest URL from the log, re-registers it, and the stale entry expires after 90 s.
  llama-server's `--api-key-file` is what protects that public URL.
* **Stable name instead:** set `LLM_PEER_URL=https://<machine>.llm-peers.garylvov.com` (and
  `LLM_PEER_CF_ACCESS=true`) after creating a named tunnel with `scripts/cf-machine.sh <machine>`
  (dry-run; the operator applies it) and installing its token as
  `~/.config/local-model-serve/tunnel-token` (0600). `llm join` then runs the named tunnel with
  `--protocol quic` and falls back to `http2`.
* `llm leave` deregisters and tears down the tunnel and the router.

**Latency caveat:** an off-cluster peer's traffic crosses Cloudflare twice (agent → Cloudflare →
gateway on login009 → Cloudflare → home rig). Measured on gpu2260 with a quick tunnel standing in
for a home rig: time to first byte went from 0.18 s direct to 0.34 s through gateway + quick tunnel;
server-side decode was unchanged (30.2 tok/s), but end-to-end wall time for a 150-token stream
varied from 3.2 to 5.3 s (vs 2.9–3.0 s direct), i.e. quick tunnels add jitter as well as delay.

**Oscar DNS quirk:** Oscar's resolver (172.20.0.12) returns nothing for `*.trycloudflare.com`
hostnames even though the public resolvers answer. The gateway therefore falls back to Cloudflare's
DNS-over-HTTPS (`https://1.1.1.1/dns-query`) for any peer host the system resolver cannot find, and
dials that IP while keeping the real `Host` header and TLS SNI.

### Just a client (agent machine)

Only `~/.config/local-model-serve/auth.env` is needed:

```bash
llm join --client        # prompts for the URL and key, writes auth.env (0600)
set -a; . ~/.config/local-model-serve/auth.env; set +a
export OPENAI_BASE_URL=$LLM_BASE_URL/v1 OPENAI_API_KEY=$LLM_API_KEY
export ANTHROPIC_BASE_URL=$LLM_BASE_URL ANTHROPIC_AUTH_TOKEN=$LLM_API_KEY
```

Both APIs are served: OpenAI `/v1/chat/completions`, `/v1/models`, and Anthropic
`/v1/messages` (llama-server implements both; tool calling needs `--jinja`, which the
presets set). On-cluster agents can point at `http://gpu2260:4000` directly and skip Cloudflare.

## The gateway

The gateway runs on a **login node** (`login009` by default), not on a compute node: it needs no
GPU, it survives Slurm jobs ending, and it is where the Cloudflare connector already runs, so the
tunnel origin can simply be `http://localhost:4000`. Peers default to `http://login009:4000`;
override with `LLM_GATEWAY_HOST`, `LLM_GATEWAY_URL` (or `LLM_GATEWAY_URL=none` to not join one).

```bash
ssh login009 'cd /oscar/data/stellex/glvov/local-model-serve && bin/llm gateway up'
```

It is a single `uvicorn` process at `nice -n 5` that polls each peer's `/models` every 10 s — no
busy loops. It lives in tmux session `llm-gateway` on that node and keeps running across jobs and
logouts, but **not across a login-node reboot**: after one, re-run the command above. A user
`@reboot` crontab entry would automate it (`crontab` is available on login009 and glvov has no
crontab today) — check CCV policy before adding one, and do not install a systemd unit.

```bash
llm passwd            # set the browser password (scrypt hash, 0600) - required before gateway up
llm gateway up        # :4000, tmux 'llm-gateway'
llm gateway status    # registered peers, their models, in-flight counts
```

* `/v1/*` needs the bearer key (`Authorization: Bearer` or `x-api-key`).
* Browser paths (`/`, `/status`) accept the key **or** a signed HttpOnly session cookie from
  `/login`; failed logins are rate-limited per `CF-Connecting-IP` and logged. Prompts are never
  logged and never appear on `/status`.
* `/` proxies llama.cpp's built-in WebUI from a peer; `/status` is a plain HTML page
  (2 s refresh) built from the heartbeats: per-GPU utilisation, VRAM, temperature, power,
  plus each peer's loaded models.
* Routing: a peer that has the model loaded, sticky per session header
  (`x-session-id`, `x-<vendor>-session-id`, Anthropic `metadata.user_id`) so llama.cpp's prompt
  cache hits, otherwise the peer with the fewest in-flight requests; if nobody has it loaded but
  a peer lists it, the request is forwarded with `?autoload=true`; otherwise a JSON 404/503.
* Peers that stop heartbeating for 90 s drop out of `/v1/models` automatically.

Cloudflare Access in front of `llm.garylvov.com` would be stronger than the password login;
it is documented but deliberately not applied here.

## Everyday commands

| command | what it does |
| --- | --- |
| `llm serve` / `llm stop` | start/stop this machine's router (tmux `llm`) |
| `llm up <model>` / `llm down <model>` | `/models/load` / `/models/unload` (fuzzy names) |
| `llm pause` / `llm resume` | unload everything (frees VRAM) / reload the same set |
| `llm ls` | router model list with status, GPUs and decode tok/s from `/metrics` |
| `llm pull <model>` | download with llama.cpp's own downloader into `models/` (resumable) |
| `llm build <variant>` / `llm builds` | build/list llama.cpp variants from `catalog/builds.yaml` |
| `llm gateway up\|down\|status`, `llm passwd` | the public gateway |
| `llm tunnel up\|down\|status` | this machine's own cloudflared |
| `llm join [--client]` | build + auth + serve + tunnel, or client-only auth |
| `llm auth init\|print-client` | keys and `auth.env` (never printed to a terminal) |

## Adding a model

1. Check `catalog/models.yaml` (or add an entry — verify the repo and sizes against the HF API,
   never invent one; mark `verified: false` if unchecked).
2. Add a section to the right preset, e.g.:

   ```ini
   [my-model]
   ; match: shortname alias
   hf-repo = unsloth/Some-Model-GGUF:UD-Q4_K_XL
   device = CUDA0,CUDA1
   ctx-size = 262144
   ```

3. `llm pull my-model && llm up my-model`.

Rules worth knowing: keep a draft model (MTP/DFlash) and an `mmproj` in **separate** presets
(they do not combine, llama.cpp #27408), and give Qwen-VL mmproj presets `image-min-tokens = 1024`.

## Cloudflare route for the gateway

`scripts/cf-route.sh` adds/updates exactly one ingress rule
`llm.garylvov.com → http://<gateway-host>:4000` on the existing remotely-managed tunnel,
preserving every other rule, plus the DNS CNAME. **It is a dry-run by default and prints the
before/after ingress diff**; `--apply` is an operator action.

```bash
scripts/cf-route.sh --service http://gpu2260:4000        # dry run, read-only GETs
```
