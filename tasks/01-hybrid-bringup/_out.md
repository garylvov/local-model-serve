# 01 — hybrid bring-up: outcome (2026-09-16, gpu2260, Slurm job 6409964)

Executor: Claude (Codex was over its usage limit). Everything below was measured on this node
unless it says otherwise. The brief was amended eight times while this ran (deliverables 10–18);
deliverable 15 ("SIMPLIFY: piggyback on llama.cpp infra") replaced most of the earlier design, so
the shape of the repo is the deliverable-15/16/17/18 shape, not the original 1–9 shape.

## Status per deliverable

| # | Deliverable | Status |
| --- | --- | --- |
| 1 | llama.cpp CUDA build + `scripts/build-llama.sh` | **DONE** — commit `7ceed8737` (master, 2026-09-16, tag b11002), `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, `-j60`. Targets: llama-server, llama-bench, llama-cli, llama-gguf-split, llama-app (the unified `llama` binary, needed for `llama download`). Build ≈4 min. |
| 2 | Models | **PARTIAL** — see "Models" below. Qwen3.8-27B (UD-Q4_K_XL, UD-Q5_K_XL, both mmproj) downloaded and serving. Qwen3.8-Flash-Next UD-Q4_K_XL, Qwen3.8-27B BF16 and Laguna-S-2.1 Q8_0 still downloading. MiniMax-M2.7 / Qwen3-Coder-30B dropped on operator instruction (partials deleted). |
| 3 | Profiles | **SUPERSEDED by 15** — `profiles/*.yaml` deleted; `presets/*.ini` (llama.cpp's own preset format) took their place: `oscar-8x`, `quad-24g`, `dual-24g`, `single-24g`, `peer-test`. |
| 4 | Node launcher | **DONE (reshaped)** — `bin/llm` (328 lines bash, `bin/lms` symlink) wrapping router mode. |
| 5 | Multi-node registry | **SUPERSEDED by 15/16** — JSON registry deleted; peers heartbeat to the gateway instead (no static list anywhere). |
| 6 | LiteLLM gateway | **DELETED by 15** — replaced by `gateway/llm_gateway.py` (338 lines, httpx+starlette in `.venv`). |
| 7 | `scripts/cf-route.sh` (dry-run) | **DONE** — dry-run output below. No Cloudflare writes were made, `--apply` was never run, no `cloudflared` was started. |
| 8 | Verification evidence | **MOSTLY DONE** — tok/s, tool calls (OpenAI + Anthropic), nvidia-smi, cf dry-run below. Missing: numbers for the big model (still downloading) and any measurement through Cloudflare (route not approved). |
| 9 | README | **DONE** — `README.md`: new Oscar node, home rig/single GPU, client-only, gateway, add-a-model, cf-route. |
| 10 | auth.env + bootstrap + auto profile | **PARTIAL/SUPERSEDED** — `~/.config/local-model-serve/auth.env` + `api-key` (0600) done, `llm auth init|print-client` done, preset auto-selection by GPU count done. `scripts/bootstrap.sh` deleted in favour of `llm join` (deliverable 11/15). The 1/2/8-GPU dry-run of the old YAML auto-selection is gone with the YAML profiles. |
| 11 | `llm join` + `POST /lms/enroll` | **PARTIAL** — `llm join` / `llm join --client` implemented; the HTTP enrollment service was explicitly dropped by deliverable 15 ("not an HTTP enrollment service"), so Cloudflare setup stays in `scripts/cf-machine.sh` (dry-run) and the tunnel token is installed by hand. |
| 12 | Streaming over Cloudflare | **DONE, and the shim proved unnecessary** — measured below: llama-server sends HTTP 200 + SSE headers immediately and `:` keepalive comments every `sse-ping-interval` (set to 15 s) **during prefill**, so the ~100 s Cloudflare idle limit is not hit. `--cache-reuse 256` is set in all presets and prompt-cache hits were observed. Sticky session routing lives in the gateway. **Not tested through Cloudflare** — the route needs operator approval first. |
| 13 | pause/resume + quad-24g | **DONE (hard pause only)** — `llm pause` / `llm resume` measured below; `profiles/quad-24g.yaml` became `presets/quad-24g.ini`. `--soft` does not exist: router mode has no "keep loaded but out of rotation" state (its `sleeping` state is idle-triggered via `--sleep-idle-seconds` and also frees memory). |
| 14 | `llm` CLI + catalog | **DONE** — `catalog/models.yaml` (verified against the HF API; entries carry `verified:`), fuzzy `; match:` names, `llm ls`, `llm pull`. `--replicas`/bin-packing was dropped by 15. |
| 15 | SIMPLIFY on llama.cpp infra | **DONE** — verification of what upstream provides is below; LiteLLM, the enroll endpoint, the streaming shim and the JSON registry were deleted. |
| 16 | One hostname, machines invisible | **DONE** — gateway + heartbeats + two-peer test below. |
| 17 | Browser chat + `/status` + password | **DONE** — login/cookie/rate-limit, `/status` from live heartbeats, llama.cpp WebUI proxied. |
| 18 | Multiple llama.cpp builds | **PARTIAL** — `catalog/builds.yaml`, `scripts/build-llama.sh <variant>`, `llm build`, `llm builds` exist. Only `master` is built. `poolside-laguna` and `glm53` are defined but **not built**; the GLM-5.3-Flash experiment was not started (146 GiB of weights, and the Qwen3.8 bring-up had to come first). |

## What upstream llama.cpp actually provides (deliverable 15 verification)

All checked in the vendored tree at `7ceed8737`:

| Feature | Exists? | Reference |
| --- | --- | --- |
| Router mode (no `-m`, child process per model) | yes | `tools/server/README.md` §"Using multiple models"; observed: router pid + one child per loaded model |
| `--models-preset <ini>` / `--models-dir` / `--models-max` / `--no-models-autoload` | yes | README lines 227–230; used by `llm serve` |
| `GET /models`, `POST /models/load`, `POST /models/unload`, `GET /models/sse` | yes | README §"GET /models" … §"POST /models/unload"; used by `llm up/down/pause/resume/ls` |
| Per-model GPU pinning inside a preset (`device = CUDA6,CUDA7`) | yes | `-dev/--device`; observed: child gets `--device CUDA6,CUDA7` |
| Built-in `-hf` download + `LLAMA_CACHE` + a **download-only** mode (`llama download`) | yes | `app/download.cpp`, `LLAMA_EXAMPLE_DOWNLOAD`; used for every download here |
| Anthropic `/v1/messages` (+ `count_tokens`) | yes | README §"Anthropic-compatible API Endpoints"; tool round-trip verified below |
| `--api-key` / `--api-key-file` | yes | README line 212–213 |
| `/metrics`, `/slots` (router: `?model=` query param) | yes | README lines 974, 1122–1126; `llm ls` reads `llamacpp:predicted_tokens_seconds` |
| `--cache-reuse` | yes | README line 221 |
| SSE: headers immediately + `:` pings while idle | yes, incl. during prefill | `tools/server/server-context.cpp:4413` (`content_type` set on first response), 4462–4478 (ping on `sse-ping-interval` timeout); `X-Accel-Buffering: no` at `server-http.cpp:549`. Measured below |
| `rpc-server` (pool GPUs across machines) | exists (`tools/rpc`), **not used** | would let one model span machines; not needed here and adds a failure domain |
| Remote llama-server upstreams in router mode | **no** | `tools/server/server-models.cpp` only spawns local children and proxies to `127.0.0.1:<child port>`. This is why `gateway/llm_gateway.py` exists (deliverable 16 preference order: native first, gateway second) |

## Measurements

### Serving (Qwen3.8-27B UD-Q5_K_XL, GPUs 6+7, `-np 4`, ctx 256k total, KV q8_0)

| measurement | value |
| --- | --- |
| model load (cold, NFS, with 4 downloads competing) | ~9 min first time; 8–10 s once the page cache is warm |
| prefill, 45,066-token prompt | **1337 tok/s** (33.7 s), same through the gateway (1324 tok/s) |
| prefill, 319-token prompt | 519 tok/s |
| decode, 1 stream | **26.2 tok/s** |
| decode, 4 concurrent streams | **17.5 tok/s per stream, 69.8 tok/s aggregate** |
| decode, UD-Q4_K_XL on GPUs 4+5 (peer test) | 29.9 tok/s single stream |
| time to first byte, streaming (short prompt) | 0.344 s direct, 0.360 s through the gateway |
| VRAM with the fast model up | GPU6 15011 MiB, GPU7 16209 MiB of 23028 MiB; all other GPUs idle (~30 MiB) |

Benchmark helper: `run/bench.sh <model> <concurrency> [url]`.

### Streaming / Cloudflare-survival evidence (deliverable 12)

45k-token prompt, `stream: true`, `sse-ping-interval = 15`:

```
ttfb=0.000169s total=34.608869s      # direct to the router
:            <- keepalive comment at ~15 s, during prefill
:            <- keepalive comment at ~30 s, during prefill
data: {"choices":[{... first token ...
```

Through the gateway: `ttfb=0.000163s total=34.938281s`, 2 keepalive comments, response headers
`content-type: text/event-stream`, `x-accel-buffering: no`, no gzip on API paths. So llama.cpp
itself keeps the connection alive during a long prefill and **no keepalive shim was added**
(deliverable 15's condition). The remaining untested link is Cloudflare itself, because the
`llm.garylvov.com` route is not applied yet.

### Tool calls (both APIs, direct to the router and through the gateway)

OpenAI `/v1/chat/completions` with `tools`:

```json
{"finish":"tool_calls","tool":{"name":"get_weather","arguments":"{\"city\":\"Providence, RI\"}"}}
```

Anthropic `/v1/messages` with `tools`:

```json
{"stop_reason":"tool_use","tool":[{"name":"get_weather","input":{"city":"Providence, RI"}}]}
```

Both were also produced through the gateway on :4000. Qwen3.8's chat template round-tripped tool
calls correctly here, contrary to the concern in llama.cpp issue #27139 (which is about Codex
specifically — not retested with Codex). Anthropic responses carry `thinking` blocks and the
usage showed `cache_read_input_tokens: 315`, i.e. the prompt cache is working.

### Gateway, two-peer test (deliverable 16)

Two routers on this node: `llm` (:8080, `qwen3.8-27b` on GPUs 6+7) and `llm2`
(:8081, preset `peer-test`, `qwen3.8-27b-q4` on GPUs 4+5), both heartbeating to the gateway.

* merged list: `GET /v1/models` → `["qwen3.8-27b","qwen3.8-27b-q4"]`
* routing: a request for `qwen3.8-27b-q4` moved peer :8081's `llamacpp:prompt_tokens_total` from
  0 → 59 (so it really went to that peer)
* streaming through the gateway for that model: `ttfb=0.33 s`, 49 SSE chunks
* dead peer: killed the :8081 router at 11:33:11; by 11:34:51 the gateway logged
  `peer http://gpu2260:8081 missed heartbeats; dropped`, `/v1/models` fell back to
  `["qwen3.8-27b"]`, and a request for the missing model returned
  `404 {"error":{"message":"model 'qwen3.8-27b-q4' is not served by any peer", ...}}` — no hang.

### Browser auth and status (deliverable 17)

```
401 POST /v1/chat/completions   (no key)
303 GET  /status                (redirect to /login)
303 GET  /                      (redirect to /login)
401 GET  /status.json           (no key/cookie)
401 POST /login  password=wrong
303 POST /login  password=<correct>   -> sets HttpOnly cookie
200 GET  /status  (1911 B, shows all 8 GPUs live, e.g. "0 | NVIDIA RTX A5000 | 0% | 234 / 23028 MiB")
200 GET  /        (llama.cpp WebUI proxied from the peer, text/html)
```

Failed logins are rate-limited (5 per 5 min per `CF-Connecting-IP`) and logged without the
password. **The password is currently a random value I generated and did not record anywhere —
the operator must run `llm passwd` to set a real one** (`LLM_PASSWORD=... llm passwd` for
non-interactive use). No prompt text is logged and `/status` contains no prompts or paths.

### pause / resume (deliverable 13)

With a 400-token streaming request in flight:

```
llm pause   -> deregistered from the gateway, unloaded qwen3.8-27b after 18 s
               VRAM after: every GPU back to ~24–40 MiB
llm resume  -> qwen3.8-27b loaded after 10 s, re-registered
```

Caveat measured: the in-flight stream was **cut** when the model unloaded (the client saw
`proxy error: Failed to read connection` after ~18 s of tokens). Router mode's unload does not
drain in-flight requests, and there is no soft pause. If graceful draining matters, stop sending
new requests first (the gateway drops the peer immediately on `llm pause`) and pause a moment later.

### Cloudflare dry-run (deliverable 7) — read-only GETs only

```
tunnel: lvov-ccv id=<uuid> status=healthy connections=8
config version: 10
---- ingress diff (before -> after)
--- before
+++ after
@@ -13,6 +13,11 @@
       "hostname": "dag.garylvov.com"
     },
     {
+      "hostname": "llm.garylvov.com",
+      "service": "http://gpu2260:4000",
+      "originRequest": {}
+    },
+    {
       "service": "http_status:404"
     }
   ],
---- DNS: would CREATE CNAME llm.garylvov.com -> <tunnel-id>.cfargotunnel.com (proxied)
---- dry-run: no changes made (rerun with --apply after operator approval)
```

`ccv`, `grove` and `dag` rules are preserved verbatim (the script refuses to proceed if any other
rule would change). `scripts/cf-machine.sh homerig` also ran dry: it found no existing
`llm-peer-homerig` tunnel, DNS record, Access app or `llm-gateway` service token, and printed the
six calls that would create them. **No PUT/POST/DELETE was issued to Cloudflare, and no
`cloudflared` process was started.** The Oscar tunnel currently shows 8 connections — worth an
operator glance, since it suggests more than one connector replica is attached to that token.

## Models

Downloaded (in `models/`, HF-cache layout, `LLAMA_CACHE`):

| file | size | state |
| --- | --- | --- |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 19.4 GiB | **serving** on GPUs 6+7 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 16.4 GiB | downloaded (single-GPU preset, used in the peer test) |
| `mmproj-F16.gguf` / `mmproj-BF16.gguf` | 0.86 / 0.87 GiB | downloaded (vision preset, untested) |

Still downloading (tmux sessions, ~100 MB/s combined earlier, slower now that fewer streams run):

| target | progress at hand-off | session |
| --- | --- | --- |
| `unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL` (103.7 GiB, 4 shards) | ~70 GiB | `lms-dl-flashnext` |
| `unsloth/Qwen3.8-27B-GGUF:BF16` (50.9 GiB, 2 shards) | ~17 GiB | `lms-dl-qwen38b` |
| `poolside/Laguna-S-2.1-GGUF:Q8_0` (119.9 GiB) + DFlash sidecar | ~27 GiB | `lms-dl-laguna` |

Resume/restart (idempotent, resumes from `.downloadInProgress` — verified: Laguna resumed at
25.5 GiB after I stopped it):

```bash
cd /oscar/data/stellex/glvov/local-model-serve
export LLAMA_CACHE=$PWD/models
V=vendor/llama.cpp/build/bin/llama
tmux new-session -d -s lms-dl-flashnext "$V download -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"
tmux new-session -d -s lms-dl-qwen38b   "$V download -hf unsloth/Qwen3.8-27B-GGUF:BF16"
tmux new-session -d -s lms-dl-laguna    "$V download -hf poolside/Laguna-S-2.1-GGUF:Q8_0 --dflash"
# or simply: bin/llm pull flashnext
```

Deleted on operator instruction: MiniMax-M2.7 (UD-Q3_K_XL + UD-Q4_K_XL partials) and
Qwen3-Coder-30B-A3B Q8_0 partials — 66 GiB removed from `models/unsloth/`, which no longer exists.
Their catalog entries stay, marked with sizes but no local files.

**Downloader choice:** llama.cpp's own `llama download` (`app/download.cpp`, `LLAMA_EXAMPLE_DOWNLOAD`)
into `LLAMA_CACHE=<repo>/models`, as instructed. It resumes, writes the HF-cache layout the router
expects, and pulls the mmproj/sidecar automatically (`--dflash`). Its one downside versus
`hf download` is a single connection per file (~50–55 MiB/s here vs ~100 MiB/s aggregate for
xet-based `hf download`), and one failure was observed (`download failed: Failed to read
connection (status: -1)` on the Laguna blob) which needed a manual restart — it resumed correctly.

**Big-model status:** the brief's original pick (MiniMax-M2) was dropped by the operator, and the
current pick, Qwen3.8-Flash-Next UD-Q4_K_XL on GPUs 0–5, is still downloading, so there are **no
tok/s numbers for a big model yet**. Once it lands: `bin/llm up flashnext` (preset section
`[qwen3.8-flash-next]`, GPUs 0–5, 4×64k ctx, KV q8_0), then `run/bench.sh qwen3.8-flash-next 1`
and `4`. Watch for issue #27886 (hallucination/vision) and check a real coding prompt before
trusting it.

## Repo shape and what was deleted

```
bin/llm (+ lms symlink)   catalog/{models,builds}.yaml   presets/*.ini
gateway/{llm_gateway.py,set_password.py}                 scripts/{build-llama.sh,cf-route.sh,cf-machine.sh}
README.md   models/ vendor/ run/ .venv/ (all gitignored)
```

Deleted when deliverable 15 landed (all were working code, replaced by llama.cpp's own machinery):

* `lib/lms.py` (≈1000-line Python CLI: profiles, registry writer, blue/green LiteLLM reloads,
  pause/resume state machine) and `bin/lms` as a script;
* `lib/front.py` (streaming keepalive shim + 503-on-paused front end) — llama-server already
  sends keepalives, proven above;
* `profiles/{hybrid,quality,speed}.yaml` (replaced by `presets/*.ini`);
* the LiteLLM venv and generated configs (`.venv` rebuilt with only httpx/starlette/uvicorn);
* `registry/` (JSON backend registry + `static.yaml`) — heartbeats replace it;
* `scripts/bootstrap.sh` — `llm join` covers it;
* old key files `~/.config/local-model-serve/{backend-key,gateway-master-key}` and the old
  `auth.env` (replaced by one shared `api-key` + `auth.env` with `LLM_API_KEY`/`LLM_BASE_URL`).

## Running right now on gpu2260 (left up, as asked)

| tmux session | what |
| --- | --- |
| `llm` | llama-server router, `0.0.0.0:8080`, preset `oscar-8x`, `qwen3.8-27b` loaded on GPUs 6+7 |
| `llm-heartbeat` | registers `http://gpu2260:8080` with the gateway every 30 s |
| `llm-gateway` | `gateway/llm_gateway.py` on `0.0.0.0:4000` (chat UI, `/status`, `/v1/*`) |
| `lms-dl-flashnext`, `lms-dl-qwen38b`, `lms-dl-laguna` | the three downloads above |

Logs: `run/gpu2260/llm/router.log`, `run/gpu2260/llm/gateway.log`, `models/logs/dl-*.log`.
Nothing else on the node was touched: no `scancel`, no other user's jobs, no slurm-dash sessions,
no `cloudflared`.

## Operator decisions applied (2026-09-16, after the first hand-off)

1. **Gateway moved to login009.** It now runs there in tmux `llm-gateway` (`0.0.0.0:4000`, one
   `uvicorn` process at `nice -n 5`, 10 s peer polling, no GPU dependency), started from the same
   shared-storage checkout. gpu2260's router registers to it (`llm-heartbeat` -> `http://login009:4000`)
   and a chat completion through `http://login009:4000` returned 200. The compute-node gateway on
   `gpu2260:4000` was stopped. `LLM_GATEWAY_HOST` (default `login009`) / `LLM_GATEWAY_URL` make the
   host configurable; `LLM_GATEWAY_URL=none` opts a machine out. It survives this job ending and any
   logout, but not a login-node reboot - README documents re-running `llm gateway up` and notes that
   a user `@reboot` crontab entry is possible (crontab exists on login009, glvov has none) subject to
   CCV policy; no systemd unit was added.
2. **Laguna abandoned.** `lms-dl-laguna` stopped and `models/models--poolside--Laguna-S-2.1-GGUF`
   (59 GiB of partials) deleted. `catalog/models.yaml` now marks it `downloaded: false` with the
   reason (DFlash rejected upstream #26669, CUDA NaN logits #27899, vendor-only benchmarks), and the
   `oscar-8x` preset section is labelled NOT DOWNLOADED.
3. **Cloudflare dry-run against the login-node origin** (`http://localhost:4000`, matching the style
   `dag.garylvov.com` already uses, because the connectors run on login009): diff below, still no writes.

```
--- before
+++ after
@@ -13,6 +13,11 @@
       "hostname": "dag.garylvov.com"
     },
     {
+      "hostname": "llm.garylvov.com",
+      "service": "http://localhost:4000",
+      "originRequest": {}
+    },
+    {
       "service": "http_status:404"
     }
   ],
```

**Tunnel connector census (read-only):** the 8 connections are **two connectors, both owned by
glvov and both running on login009** - pids with 36d and 20d uptime, matching the API's
`run_at` 2026-08-10T22:23Z (tmux `cloudflared-setup`) and 2026-08-26T19:42Z (tmux `slurm-dash`),
4 QUIC connections each (colos ewr01/05/08/12 and ewr01/11/12/15), cloudflared 2026.7.3. A third
cloudflared on login009 belongs to another user (yma158) on a different tunnel. Both glvov
connectors serving one token is why `http://localhost:4000` is safe here - but if a connector is
ever started on another host, `localhost` would resolve on that host instead, so either keep both
connectors on login009 or change the origin to `http://login009:4000`.

## One-command node attach (added after the operator's second round)

`llm join` is now the whole flow and takes no arguments: it checks the GPUs, builds llama.cpp only
if the pinned build is missing, picks the preset by GPU count, starts the router (whose preset marks
one model `load-on-startup = true`, so it loads itself), starts the 30 s heartbeat to
`http://$LLM_GATEWAY_HOST:4000` (default `login009`), and prints the model table. `llm leave`
deregisters and stops heartbeat, tunnel and router.

Tested on gpu2260 by tearing the node down and bringing it back:

```
$ llm leave      -> router stopped, gateway /peers == []   (all GPUs back to ~30 MiB)
$ llm join       -> llama.cpp already built (7ceed8737)
                    preset oscar-8x.ini (8 GPUs)
                    router up on 0.0.0.0:8080
                    heartbeating http://gpu2260:8080 to http://login009:4000
                    qwen3.8-27b loading (auto, load-on-startup)   [1.4 s wall for the command]
$ llm join       -> idempotent: "router already running", re-registers, prints the table
gateway: /v1/models == ["qwen3.8-27b"], chat completion through login009 == 200
```

One bug found and fixed on the way: `--models-max 0` (unlimited) is rejected when any preset uses
`load-on-startup` ("number of models to load on startup (1) exceeds models_max (0)"), so the router
now runs with `--models-max ${LLM_MODELS_MAX:-8}`.

Abrupt job death is covered by the same path as the earlier dead-peer test: heartbeat TTL 90 s with
a 10 s poll, measured drop 100 s after a hard kill, and `/v1/models` stops advertising the models.
A second GPU node was not allocated (no idle gpu nodes in `sinfo` at the time, and allocating one
was outside what the brief allows), so the test was the leave/join cycle on gpu2260.

README documents the copy-paste lines, including `ssh <node> 'cd <repo> && bin/llm join'` (preferred,
survives the step) and the `srun --overlap --jobid <id> --pty bash` alternative.

## Needs an operator decision

1. **Cloudflare route: approved by the operator, but BLOCKED on this machine.** The intended call is
   `scripts/cf-route.sh --service http://login009:4000 --apply` (host-qualified origin, per the
   operator). I re-checked the live config first: version 10, ingress still exactly
   ccv (unix socket) / grove (http://gpu3201:8874) / dag (http://localhost:8501) / http_status:404,
   saved to `run/ingress-before.json` for rollback. The `--apply` run was then refused by the Claude
   Code permission layer ("DNS / Domain / Cert Changes"), which only the human operator can grant, so
   **no write was made and nothing at `https://llm.garylvov.com` answers yet**. Either allow that
   Bash action and re-run the one command, or run it yourself; afterwards the checks to run are
   `curl -o /dev/null -w '%{http_code}' https://llm.garylvov.com/v1/models` (401 without a key,
   200 with), `https://ccv.garylvov.com/api/health` (must stay 200), and a streaming completion for
   TTFB/tok/s through Cloudflare versus direct.
2. **Set the browser password**: `llm passwd` (mine is random and unknown). Optionally put
   Cloudflare Access in front of `llm.garylvov.com` instead — stronger, and documented, not applied.
3. **Big model**: confirm Qwen3.8-Flash-Next UD-Q4_K_XL as the GPUs 0–5 model once downloaded, or
   pick MiniMax-M2.7 / Laguna / GLM-5.3-Flash (needs the `glm53` PR build) instead.
4. ~~Laguna~~ — decided: abandoned and deleted (see above).
5. **GLM-5.3-Flash experiment** (deliverable 18) not started: it needs a PR build plus 146 GiB.
6. Two small known warts: `dedup-cache-models = true` does not hide the raw cache entries for
   unsloth repos (their cache tag is `Q4_K_XL` while the preset asks for `UD-Q4_K_XL`), so
   `llm ls` shows a couple of duplicate `unsloth/...` rows; and the vision preset
   (`qwen3.8-27b-vision`, `--image-min-tokens 1024`) has never been loaded or given an image.
