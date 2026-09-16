# 01 — hybrid bring-up of local-model-serve

**Every claim in this brief is a LEAD TO VERIFY.** Measure at the point of use
(`nvidia-smi`, `df`, `free`, `hostname`, HF file listings) before relying on any
number here. **No background waits:** never `sleep`/poll-loop waiting for a
download, build, or server. Start long work in tmux, check once, and if it is not
done, record status in `_out.md` and exit.

## Goal

Build a reusable repo at `/oscar/data/stellex/glvov/local-model-serve` (already
created, not yet a git repo — `git init` it) that serves local LLMs for agentic
coding with llama.cpp, fronted by one OpenAI- and Anthropic-compatible gateway, so
that agents can later call `https://llm.garylvov.com` like any remote API. It must
be trivial to add more nodes (other Oscar GPU nodes, or the operator's home rig)
to add capacity. First target profile: **hybrid** on this node.

## Environment (verify)

- Node `gpu2260`, inside Slurm job 6409964. 8× RTX A5000 (~22.5 GiB usable each).
  NVLink pairs 0-1, 2-3, 4-5, 6-7; GPUs 0-3 on NUMA 0 (cores 0-29), 4-7 on NUMA 1
  (cores 32-61). ~1 TB RAM, 2× EPYC 7532.
- CUDA module `cuda/12.9.0-cinr` (nvcc 12.9 on PATH). `module load
  llama-cpp/7158-c32l` is an old (≈Nov 2025) CUDA build — use only as fallback.
- `hf` CLI at `~/.local/bin/hf`; `pixi` available; `cmake` at /usr/bin and as modules.
- Storage: model weights → `<repo>/models/` (gitignored; override via `LMS_MODELS_DIR`).
  Operator amendment 2026-09-16 — supersedes the earlier `/oscar/data/stellex/glvov/models`.
  **Do not put weights or venvs in `$HOME`** (home quota nearly full).

## Deliverables

1. **llama.cpp**: clone latest master to `/oscar/data/stellex/glvov/local-model-serve/vendor/llama.cpp`
   (gitignored) and build with CUDA (`GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`,
   Release, `-j` using this job's cores). Record the commit hash. A `scripts/build-llama.sh`
   must reproduce this on any Linux+CUDA box (arch configurable).
2. **Models** (download with `hf download`, only the needed quant shards):
   - Big: the newest MiniMax-M2-series GGUF that current llama.cpp supports (check
     unsloth/bartowski on HF). Place on GPUs 0-5 (~135 GiB). Pick the largest quant
     where weights + KV for `--parallel 4` at ≥64k tokens/slot (q8_0 KV) fit on
     those 6 GPUs; if only a Q4 fits with a few MoE layers on CPU (`--n-cpu-moe`),
     compare that against the best all-GPU quant with `llama-bench` and pick by
     measured decode tok/s while keeping quality ≥ Q3_K_XL-class. Justify in `_out.md`.
   - Fast: Qwen3-Coder-30B-A3B-Instruct GGUF (Q8_0 if it fits with large context,
     else Q6_K) on GPUs 6-7.
   If a clearly better agentic-coding model exists in the same size class and is
   supported by llama.cpp, note it in `_out.md` as a recommendation — do not swap
   without operator approval.
3. **Profiles**: declarative files (e.g. `profiles/hybrid.yaml` or `.env`) listing
   instances: model path/HF repo+file, GPUs, NUMA node, ctx, parallel, KV type,
   extra llama-server args, port, public model alias (`minimax-m2`, `qwen3-coder-30b`).
   Include at least also `profiles/quality.yaml` (one big model on all 8) and
   `profiles/speed.yaml` (4× fast model, one per NVLink pair) — these need not be
   downloaded/tested now.
   Required llama-server flags: `--jinja` (tool calling), `-fa on`, `-ngl 99`,
   `--metrics`, `--api-key-file` (shared backend key), numactl pinning, bind
   `0.0.0.0` (gateway may be on another node — key is mandatory since cluster
   network is shared).
4. **Node launcher** `bin/lms` (bash, or a small Python CLI if cleaner): `lms up <profile>`,
   `lms down`, `lms status`, `lms logs <instance>`. One tmux session per instance
   (`lms-<instance>`), logs under `/oscar/data/stellex/glvov/local-model-serve/run/<host>/`
   (gitignored). Idempotent: refuses to double-start. Works on any node — no
   hardcoded hostnames/GPU counts outside profiles.
5. **Multi-node registry**: on healthy start, a node writes
   `registry/<host>-<instance>.json` (url `http://<host>:<port>`, model alias,
   profile, ts) on the shared FS; `lms down` removes it. For machines off the
   shared FS (home rig), a checked-in-template `registry/static.yaml` (real file
   gitignored) lists extra backend URLs + key refs. Document how the home rig
   exposes its backend (e.g. its own Cloudflare route or Tailscale) — no implementation needed.
6. **Gateway**: LiteLLM proxy (install in a venv/pixi env under the repo, gitignored)
   on port 4000 on this node, config generated from the registry by `lms gateway
   up|reload|down`. Must: load-balance across backends sharing an alias; expose
   `/v1/chat/completions`, `/v1/models`, and Anthropic `/v1/messages`; require a
   master key. Keys live in `~/.config/local-model-serve/` mode 0600 (generate
   them; never print or commit values).
   If LiteLLM proves unworkable, justify an alternative in `_out.md`.
7. **Cloudflare**: the existing remotely-managed tunnel (see
   `/oscar/data/stellex/glvov/slurm-dash/PROGRESS.md` sections "Tunnel operation" and
   "Cloudflare, Access, and rate limiting"; `deploy/run-tunnel.sh`) already routes
   `ccv.`, `grove.`, `dag.garylvov.com`. Write `scripts/cf-route.sh` that adds/updates
   ONE ingress rule `llm.garylvov.com → http://<gateway-host>:4000` while preserving
   every other rule verbatim, plus the DNS CNAME if missing. **Default is dry-run
   (prints the before/after ingress diff). `--apply` exists but you must NOT run it.**
   Read-only GETs against the Cloudflare API using
   `~/.config/slurm-dash/cloudflare-api-token` are allowed to validate the dry-run.
   **Hard prohibitions:** do not PUT/POST/DELETE anything on Cloudflare; do not start
   any `cloudflared` process (a second connector on the same token breaks the unix-socket
   route for ccv); do not read or print token values beyond passing them to curl.
8. **Verification (real evidence in `_out.md`)**:
   - `llama-bench` or server-timing numbers: prompt-processing and decode tok/s for
     each instance, single stream and 4 concurrent.
   - A real tool-call round trip through the gateway for each alias, via both the
     OpenAI (`tools`) and Anthropic (`/v1/messages` with `tools`) APIs, showing a
     well-formed tool call is returned.
   - `nvidia-smi` memory per GPU with hybrid up.
   - The cf-route dry-run output (with any secrets redacted).
9. **README.md**: quick start on a new node / home rig, profile format, add-a-node
   flow, how agents configure (OpenAI base URL `https://llm.garylvov.com/v1`,
   Anthropic base URL `https://llm.garylvov.com`, key), and the cf-route step.

10. **Auth file + reproducible multi-machine setup** (operator amendment 2026-09-16):
   - **Client auth file:** `~/.config/local-model-serve/auth.env`, mode 0600. It holds
     `LMS_BASE_URL`, `LMS_OPENAI_BASE_URL`, `LMS_ANTHROPIC_BASE_URL` and `LMS_API_KEY`.
     `lms auth init` generates it, and `lms auth print-client --host <name>` prints a
     copy for another machine without echoing the key into logs. Document how agents
     use it: `source` the file, set `OPENAI_BASE_URL`/`OPENAI_API_KEY`, and set
     `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`.
   - **One-command bootstrap on any Linux+NVIDIA machine:** `scripts/bootstrap.sh`
     clones or updates the repo, builds llama.cpp for the detected GPU architecture,
     and creates the gateway env if that machine is the gateway. It must be
     idempotent and must not depend on Slurm or modules; use them only when present.
   - **Auto profile:** `lms up auto` reads the GPU count and free VRAM from
     `nvidia-smi` and picks a fitting profile. Ship `profiles/single-24g.yaml` (one
     24 GB GPU, the fast Qwen3-Coder-30B-A3B at a quant that fits with a useful
     context) and a `profiles/single-*` pattern for other VRAM sizes. On boxes with
     lots of RAM, allow `--n-cpu-moe` spillover. Dry-run the auto-selection on this
     node for 1, 2 and 8 GPUs (use `CUDA_VISIBLE_DEVICES` or a flag that fakes the
     GPU inventory) and paste the output into `_out.md`.
   - **Off-cluster machines join the gateway through Cloudflare:** each machine
     runs its OWN tunnel connector, with its own tunnel and its own token file at
     `~/.config/local-model-serve/tunnel-token` (0600). It publishes
     `<machine>-llm.garylvov.com` pointing at its local llama-server. That hostname
     is protected by a Cloudflare Access application that accepts one Access service
     token. The gateway stores the service token's id/secret in
     `~/.config/local-model-serve/cf-access.env` (0600) and sends the
     `CF-Access-Client-Id` and `CF-Access-Client-Secret` headers on those backend
     entries in `registry/static.yaml`. Write:
     - `scripts/cf-machine.sh <machine>`: a dry-run by default that prints the API
       calls it would make (create tunnel, DNS CNAME, ingress, Access app and policy,
       service token).
     - `bin/lms tunnel up|down|status`: runs cloudflared for THIS machine's own token
       in tmux session `lms-tunnel`.

     Implement these but do not run `--apply` and do not start `lms tunnel up` here.
     The existing Oscar tunnel token must never be reused on another machine.
   - README gets a section for each case: "new Oscar node", "home rig / single GPU",
     and "just a client agent" (only needs auth.env).

11. **One command per machine** (operator amendment 2026-09-16, supersedes the
   multi-step flow in deliverable 10):
   - **Server:** `curl -fsSL <raw bootstrap URL> | bash -s -- join`, or `./lms join`
     from a checkout. It does everything: clone/update, build llama.cpp, auto-pick
     the profile, start the servers, get this machine its own tunnel + Access
     protection, start the tunnel, and register with the gateway.
   - **Client:** `lms join --client` just writes `auth.env`.
   - **Its only input** is one enrollment file,
     `~/.config/local-model-serve/join.env` (0600), holding the gateway URL and an
     enroll key. A join prompts for these two values if the file is missing.
   - **The Cloudflare API token never leaves the gateway host.** The gateway exposes
     `POST /lms/enroll`, authenticated with the enroll key and rate-limited. The
     gateway side then:
     - creates or updates the machine's tunnel, DNS record and Access application
       through the CF API (the code path behind `cf-machine.sh`);
     - returns that machine's tunnel token;
     - adds the backend entry, with the Access service-token headers, to the
       registry and reloads LiteLLM.
     `lms leave` undoes all of this.
   - **Guardrail:** implement the enroll endpoint with a `LMS_CF_DRY_RUN=1` default,
     test it end to end in dry-run mode on this node, and do not enable real CF
     writes.

12. **Streaming over Cloudflare** (operator amendment 2026-09-16):
   - **Keep the stream flowing.** Cloudflare's proxy returns a 524 if the origin is
     silent for about 100 s, and a long agent prompt can take longer than that to
     prefill before the first token. Make the gateway send HTTP 200 + SSE headers
     immediately and emit `: keepalive` SSE comments every ~15 s until the first
     token. Verify whether LiteLLM or llama-server already do this; if neither does,
     add a thin streaming shim in front of LiteLLM. Test this with a synthetic
     slow-prefill request longer than 100 s and show it survives.
   - **Response headers on streams:** `Content-Type: text/event-stream`,
     `Cache-Control: no-cache, no-transform`, `X-Accel-Buffering: no`, no gzip.
   - **Cut prefill with the prompt cache.** Agents resend a growing context every
     turn, so tune llama-server for it: `--cache-reuse 256`, keep the slot prompt
     cache, and use sticky routing in LiteLLM so the same conversation hits the same
     backend slot (e.g. affinity on a session/user header). Measure the time to first
     token on a repeated long context with and without these settings.
   - **Tunnel connectors:** use `--protocol quic` and fall back to `http2` if UDP is
     blocked (detect this and record which one worked). Document a second connector
     replica per machine token as the HA option; it is safe only for that machine's
     own tunnel.
   - **Direct path on the cluster:** when `gpu2260:4000` is reachable,
     `lms auth init` on an Oscar node writes the direct URL, so on-cluster agents skip
     Cloudflare.
   - **Put numbers in `_out.md`:** direct-to-llama-server vs through the gateway,
     giving time to first token and decode tok/s for each. The measurement through
     Cloudflare waits until the route is approved.

13. **Pause/resume + 4-GPU home rig** (operator amendment 2026-09-16):
   - `lms pause [instance|--all] [--soft] [--timeout 60s]`
     - Default (hard) pause:
       1. Mark the backend(s) `paused` in the registry and reload the gateway, so no new requests are routed to them.
       2. Drain requests already in flight: poll `/slots` or `/metrics` until they are idle, or until the timeout.
       3. Stop the llama-server processes so the VRAM is freed.
     - `--soft`: take the backends out of rotation but keep the model loaded, so resuming is instant.
     - Run on the gateway host, `lms pause --all` pauses every node, including remote ones (they are removed from routing).
   - `lms resume [instance|--all]`: restart the servers from the recorded profile, wait for `/health`, then put them back in rotation.
   - `lms status`: show a state column (`up` / `soft-paused` / `paused` / `down`) for every backend across all nodes.
   - When a model alias has no active backends, the gateway returns a clear 503 JSON error ("all backends for X paused"); it must not hang.
   - Registry state must survive a gateway restart.
   - Profile `profiles/quad-24g.yaml` for a 4× 24 GB machine: MiniMax-M2 at a quant that fits 4 GPUs, with `--n-cpu-moe` spillover when RAM allows. Record the RAM needed, and the fallback profile to use when RAM is not enough. `lms up auto` must choose this profile for 4 GPUs; add 4 GPUs to the auto-selection dry-run.
   - Test on this node: soft pause, hard pause, and resume of the fast instance, each while a streaming request is in flight. Put timings and the resulting VRAM state in `_out.md`.

14. **`llm` CLI + model catalog** (operator amendment 2026-09-16):
   - **Rename the CLI to `llm`.** It is not on PATH on Oscar today, but a home rig may have Simon Willison's `llm`, so detect that and warn. Keep `lms` as a symlink. Every earlier `lms …` command in this brief becomes `llm …`.
   - **Model catalog** `catalog/models.yaml`, checked into the repo.
     - Each entry has:
       - a short name plus aliases and fuzzy-match tokens, so `minimax`, `m2`, `qwen-coder`, `qwen3-coder-30b` and typos resolve;
       - the HF repo and a quant ladder, each rung with file pattern, size on disk, and VRAM needed at a given ctx/parallel;
       - chat-template notes, recommended sampling, and default llama-server args;
       - capability tags (tools, reasoning, fim, vision);
       - an optional draft model for speculative decoding.
     - Seed it with at least: MiniMax-M2 (newest supported), Qwen3-Coder-30B-A3B, gpt-oss-120b, gpt-oss-20b, GLM-4.x-Air, Devstral-Small-2 / Devstral-2, and the newest dense ~27-32B Qwen coder/instruct model that llama.cpp supports.
     - Verify every HF repo and file actually exists (read-only HF API calls). Mark unverified entries `verified: false`; never invent repos.
   - **`llm up <model> [--quant Q] [--gpus 0,1] [--ctx N] [--parallel N] [--replicas N]`:**
     - resolve the name, download it if missing (resumable, into `models/`);
     - pick GPUs automatically: prefer free GPUs, prefer NVLink pairs and one NUMA node, use as few GPUs as fit;
     - choose the best quant that fits, or `--n-cpu-moe` spillover if allowed;
     - start the server, register the backend under the catalog name, and reload the gateway;
     - `--replicas N` packs N copies onto separate GPU sets.
     - Print what it chose and why.
   - **`llm down <model|instance>`** and **`llm ls`**:
     - `llm ls` shows running models across all nodes, with node, GPUs, quant, state and tok/s from `/metrics`;
     - `llm ls --catalog` shows what can run on this machine and which quant would fit;
     - `llm pull <model>` only downloads.
   - **Profiles** become thin lists of `llm up` specs, so `llm up hybrid` still works, and `auto` uses the catalog.
   - **Two separate registries:**
     - the model **catalog** (what can run; in git);
     - the backend **registry** (what is running where; runtime, gitignored).
     The gateway's `/v1/models` lists only the aliases that have running, unpaused backends.
   - **Tests:** on this node, `llm up qwen-coder --gpus 6,7`, then `llm down`, then `llm up qwen-coder --replicas 2`, and check the gateway balances across both replicas. Paste the outputs into `_out.md`.

15. **SIMPLIFY: piggyback on llama.cpp infra** (operator amendment 2026-09-16 — this
   OVERRIDES any conflicting part of deliverables 3–14.)
   Principle: use what llama.cpp already provides; write only thin glue. Keep the repo
   small enough to read in ten minutes.

   **Before building anything more, verify in current upstream llama.cpp master:**
   - llama-server **router mode**: running with no `-m`, plus `--models-dir` /
     `--models-preset <ini>`, spawns one child process per model and exposes
     `/models`, `/models/load` and `/models/unload`. Check `tools/server/README.md`.
   - built-in `-hf` download and cache (`LLAMA_CACHE`);
   - Anthropic `/v1/messages` compatibility;
   - `--api-key` / `--api-key-file`;
   - `/metrics`, `/slots`;
   - `--cache-reuse`;
   - SSE streaming headers, and whether the server sends anything before the first token;
   - `rpc-server`, for pooling GPUs across machines.
   Record what exists, with the doc or source reference, in `_out.md`.

   **Target design, adjusted to what exists:**
   - **One llama-server router per machine.** It runs on port 8080 with an API key file,
     inside tmux session `llm`. Its models come from a preset INI generated from
     `catalog/models.yaml`, or the catalog *is* the INI if that is simpler. Each preset
     carries GPU assignment (`CUDA_VISIBLE_DEVICES`/`-dev`/`-ts`), quant, ctx and args.
     Choosing the fork binary for Laguna DFlash is a per-preset setting only if router
     mode allows it; otherwise run that one model as a plain llama-server.
   - **`llm`** is a short bash script (well under ~300 lines) wrapping the router and curl:
     - `llm up <model>` → router `/models/load`
     - `llm down <model>` → `/models/unload`
     - `llm pause` → unload all
     - `llm resume` → reload the last set
     - `llm ls` → `/models`
     - `llm pull <model>` → llama.cpp `-hf` download
     - `llm serve` → start the router
     - `llm tunnel` → start this machine's cloudflared
     - `llm join` → build llama.cpp, write config, serve, tunnel
     Fuzzy name matching is fine if it is a few lines.
   - **Auto GPU fit:** keep it simple. A preset may list per-GPU-count variants (e.g.
     `single-24g`, `quad-24g`, `oscar-8x`), and `llm serve` picks one by `nvidia-smi` count.
     No bin-packing solver.
   - **DROP:** the LiteLLM gateway, the Python enroll endpoint, the custom streaming shim,
     and the separate JSON backend registry.
     - Only re-add a keepalive shim if a test shows Cloudflare actually kills a real
       long-prefill stream from llama-server.
     - Pause and "all paused" become router behaviour: unloaded models are simply absent
       or return llama-server's own error.
   - **Multi-machine:** each machine is its own router behind its own Cloudflare
     hostname (`llm.garylvov.com` for Oscar, `<machine>-llm.garylvov.com` elsewhere), with
     the same API key in `auth.env`.
     - Cross-machine routing stays out of scope for now; document that agents pick a
       hostname.
     - If llama.cpp itself offers multi-backend routing, or `rpc-server` fits, document how.
     - Tunnel and DNS setup per machine is a documented one-time step or a dry-run
       `scripts/cf-machine.sh`. It is not an HTTP enrollment service.
   - **Auth:** one `~/.config/local-model-serve/auth.env` (0600) holding `LLM_API_KEY` and
     `LLM_BASE_URL`. llama-server enforces the key. Optionally put Cloudflare Access in
     front (document only).
   - **What survives:** work already done that fits this design (build script, catalog,
     downloads, measurements).
   - **Delete:** code that doesn't fit, rather than leaving it dormant. List what you
     deleted in `_out.md`.
   - **Verification:** `llm serve` on this node, then `llm up` for the fast model and for
     Laguna, then a tool-call round trip via OpenAI and Anthropic endpoints directly
     against the router, then `llm pause` / `llm resume`, plus tok/s numbers.

16. **One hostname, machines invisible to agents** (operator amendment 2026-09-16;
   overrides 15's "cross-machine routing out of scope").
   Agents only know `https://llm.garylvov.com` plus the key. Any machine that runs
   `llm join` adds its models automatically, and agents never name a machine.

   **Order of preference — use the first one that works:**
   1. **llama.cpp native.** Check whether the llama-server router can use remote
      llama-server instances as upstreams, or whether any upstream llama.cpp tool does
      multi-host model routing. If it can, use it.
   2. **Otherwise, a tiny gateway `gateway/llm_gateway.py`** (one file, ≤ ~200 lines).
      Use the stdlib plus at most aiohttp or httpx+starlette, in a venv under the repo.
      It lives in the same repo and starts with `llm gateway`. It:
      - accepts OpenAI `/v1/*` and Anthropic `/v1/messages`, reads `model` from the JSON
        body, and streams the upstream response back byte for byte (no buffering, no
        rewriting);
      - keeps a peer list; every ~10 s it polls each peer's `/models` (or `/v1/models`)
        with the shared key;
      - routes to a healthy peer that has the model loaded, picking the one with the
        fewest in-flight requests; otherwise a peer that can load it; otherwise a clear
        404/503 JSON error;
      - makes the merged model list the answer to `GET /v1/models`;
      - pins a session to a peer when a stable conversation key header exists, so the
        prompt cache hits;
      - has a `POST /peers/register` endpoint (same API key), which `llm join` / `llm serve`
        on a peer calls every ~30 s as a heartbeat carrying its own URL. Peers that miss
        heartbeats drop out, and `llm pause` deregisters. No static list of rigs anywhere.
      - The gateway also counts as a peer when the gateway machine serves models.
   3. **Peer reachability:**
      - Oscar nodes: the gateway reaches peers directly at `http://<host>:8080`.
      - Off-cluster machines: each runs its own cloudflared on a dedicated tunnel,
        publishing `<machine>.llm-peers.garylvov.com` (or a similar name the gateway never
        exposes to agents), protected by a Cloudflare Access service token. The peer
        registers that URL, and the gateway adds the CF-Access headers from
        `~/.config/local-model-serve/cf-access.env`.
      - Tunnel creation stays a dry-run script (`scripts/cf-machine.sh`). Never apply it,
        never reuse the Oscar slurm-dash tunnel token.
      - Document one caveat: plain multi-connector Cloudflare load balancing on one
        hostname is NOT model-aware and prefers the nearest connector, which is why the
        gateway exists.
   4. **Tests on this node:**
      - Start 2 routers as fake "peers" on different ports/GPUs, with a different model
        on each.
      - Register both, and show `/v1/models` merges them.
      - Show a request for each model reaches the right peer, streaming works through
        the gateway, and killing a peer drops it within the heartbeat timeout.
      - Show the tool-call round trip via OpenAI and Anthropic through the gateway.

17. **Browser UI, live status, and auth** (operator amendment 2026-09-16):
   - **Chat UI.** Do not write one. Serve llama.cpp's built-in WebUI through the gateway,
     reverse-proxying `/` and the UI's asset paths to a peer, with the model picker
     driven by the merged `/v1/models`. If proxying the built-in UI turns out to be
     awkward, say so in `_out.md` and serve a minimal single-page chat that talks to
     `/v1/chat/completions` (streaming, model dropdown, ≤ ~150 lines) instead.
   - **Status page at `/status`** (and `/status.json`), rendered by the gateway from data
     the peers already send. Extend the heartbeat so each peer reports:
     - hostname, GPU count, and per-GPU name, utilization %, VRAM used/total,
       temperature and power, collected from `nvidia-smi --query-gpu=... --format=csv`
       (parse it; do not add a Python NVML dependency);
     - loaded models with their GPU assignment, ctx and state;
     - llama-server `/metrics` counters: tokens/s, requests in flight, KV cache use;
     - uptime and the llama.cpp build.
     Keep the page plain HTML with a ~2 s auto-refresh: one table per machine and one row
     per GPU. No JS framework, no build step.
   - **Auth — two doors, one hostname:**
     - `/v1/*` (agents): `Authorization: Bearer <LLM_API_KEY>`, as now.
     - Browser paths (`/`, `/status`, UI assets): a password gate. Implement a small
       login form that sets a signed, HttpOnly, Secure cookie (HMAC over user + expiry,
       with the secret in `~/.config/local-model-serve/auth.env`). Store the password as a
       hash (scrypt or bcrypt via stdlib `hashlib.scrypt`), never in plaintext, and never
       commit or print it. `llm passwd` sets it.
     - A bearer key must also be accepted on browser paths, so curl and scripts still work.
     - Rate-limit failed logins, and log auth failures with the source IP from
       `CF-Connecting-IP`.
     - **Document** (do not apply) putting Cloudflare Access in front of the browser paths
       as the stronger option, with the API paths bypassed via a service token.
   - **Why this matters:** the key is the only thing between the public internet and a
     model that agents run with filesystem access. So: no default password (the first
     `llm serve` without one refuses to expose browser paths), TLS only through Cloudflare,
     never log prompt content, and `/status` must not leak prompts or file paths beyond
     model names.
   - **Test:** unauthenticated requests to `/v1/chat/completions`, `/status` and `/` are
     rejected; the right key and the right password get through; the status page shows
     live GPU numbers from this node; the chat UI streams a reply.

18. **Multiple llama.cpp builds, chosen per model** (operator amendment 2026-09-16):
   - `scripts/build-llama.sh <variant>` builds into `vendor/llama.cpp-<variant>/` from a
     variant spec in `catalog/builds.yaml`: git remote, ref (branch, tag or PR ref such as
     `refs/pull/28243/head`), and cmake flags. Ship variants `master` (pinned to a known-good
     commit), `poolside-laguna` (poolsideai/llama.cpp branch `laguna`) and `glm53`
     (whichever open GLM-5.3 PR looks most alive — check its CI and recent comments).
   - A catalog model entry may name a `build:`. `llm up` starts that model with that
     variant's binary. If router mode cannot vary the binary per model, run such a model as
     a standalone llama-server on its own port and register it as a local peer, which the
     gateway already handles.
   - `llm build <variant>` builds or rebuilds one; `llm builds` lists variants with their
     commit and build date. Builds are gitignored and disposable.
   - Record for each variant: build time, disk use, and whether the model loaded.
   - **GLM-5.3-Flash experiment** (only after the Qwen3.8 bring-up works): build the `glm53`
     variant, fetch `unsloth/GLM-5.3-Flash-GGUF` UD-IQ4_XS (about 146.1 GiB) if disk allows,
     and try to load it on all 8 GPUs. Check GGUFs against the PR they were produced for,
     since an unmerged converter may have changed. Watch for the CUDA output corruption in
     issue #26027 for the GLM_DSA architecture: run a short generation and check for garbage
     or NaN before benchmarking. Report go/no-go in `_out.md`; don't make it the default.

## Constraints

- Do not touch other users' jobs, `scancel`, or the slurm-dash repo/tmux sessions.
- Leave hybrid + gateway running in tmux on gpu2260 when done.
- Commit in logical steps (no secrets, no weights, no vendor/run dirs). Commit
  message trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- If downloads are not finished when everything else is ready, stop: write what
  remains and the exact resume commands in `_out.md`.

## Output

Write `tasks/01-hybrid-bringup/_out.md`: status per deliverable (DONE / PARTIAL /
BLOCKED with reason), chosen model files + sizes, llama.cpp commit, measured
numbers, tmux sessions running, deviations from this brief, open questions.
