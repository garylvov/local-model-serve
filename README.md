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

### New Oscar node (or any Linux + NVIDIA machine)

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

### Home rig / single GPU

Same as above; `llm serve` picks `presets/single-24g.ini` (or `quad-24g` on 4 GPUs) from the
GPU count. Off-cluster machines are reached through **their own** Cloudflare tunnel:

```bash
scripts/cf-machine.sh homerig            # DRY RUN: prints the tunnel/DNS/Access calls to make
# after the operator applies them by hand, install that machine's own token:
install -m 600 <token> ~/.config/local-model-serve/tunnel-token
llm tunnel up                            # cloudflared --protocol quic, auto-falls back to http2
# and in auth.env:  LLM_PEER_URL=https://homerig.llm-peers.garylvov.com  LLM_PEER_CF_ACCESS=true
```

The peer hostname is protected by a Cloudflare Access service token; the gateway sends the
`CF-Access-Client-Id/Secret` headers from `~/.config/local-model-serve/cf-access.env`.
Agents never see that hostname. **Never reuse the slurm-dash tunnel token** — a second
connector on it breaks the existing `ccv`/`grove`/`dag` routes.

*Caveat:* you could publish several connectors on one hostname and let Cloudflare load-balance,
but that is not model-aware (and prefers the nearest connector), which is exactly why the
gateway exists.

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
