# Local agentic-coding model research: llama.cpp on 8x RTX A5000

Research date: 2026-09-16. llama.cpp master was checked at commit `7ceed873` (latest release tag b11002, 2026-09-16; https://github.com/ggml-org/llama.cpp/releases).
Every row cites its source. **V** marks a vendor-reported number and **I** an independent one. "Not found" means I searched and found nothing.

**Measured hardware:** `nvidia-smi` on this node shows `NVIDIA RTX A5000, 23028 MiB`, which is 22.49 GiB per GPU, not 24. Weight budgets, after about 2 GiB for compute buffers:
- (a) 8 GPUs: 179.9 GiB, minus 25 GB of KV, gives **≤154 GiB**.
- (b) 6 GPUs: 134.9 GiB, minus 25 GB, gives **≤110 GiB**.
- (c) 1 GPU: 22.49 GiB, minus 4 GB of KV and about 0.5 GiB of CUDA context, gives **≤18.3 GiB**.

Vision models must also fit their mmproj in these budgets.

## Task 1: llama.cpp support in master

Sources for "arch in master": the converter registrations in `conversion/*.py` and the arch list in `src/llama-arch.cpp` (https://github.com/ggml-org/llama.cpp/tree/master/conversion, https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp).

Tool-parser routing comes from `common_chat_try_specialized_template` in https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp. Templates without a specialized handler use the template-derived autoparser. docs/function-calling.md is stale and lists none of these models.

MTP support means the model has a `graph_mtp` in `src/models/*.cpp`. DFlash and EAGLE3 are generic draft types (https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md). The README no longer has a supported-models list, and docs/multimodal.md lists none of these models.

| Model (HF arch) | Arch in master (added by) | Vision (mmproj) | Tool calling | MTP / spec decoding | Known issues |
|---|---|---|---|---|---|
| Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) | Yes, `qwen35`: [#19435](https://github.com/ggml-org/llama.cpp/pull/19435), [#19468](https://github.com/ggml-org/llama.cpp/pull/19468) (merged Feb 2026) | Yes, `qwen3vl_merger` (conversion/qwen3vl.py) | Specialized Qwen3-Coder XML parser (the template has `<tool_call><function=><parameter=>`, [template](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja)) | MTP yes: [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (merged 2026-05-16); unsloth ships an MTP GGUF. DFlash2 drafter [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). EAGLE3 [#24593](https://github.com/ggml-org/llama.cpp/pull/24593) | Codex fails with the Qwen3.8 template [#27139](https://github.com/ggml-org/llama.cpp/issues/27139); DFlash2 plus `-sm tensor` aborts [#27829](https://github.com/ggml-org/llama.cpp/issues/27829); spec decoding emits an OOB token on Vulkan [#28158](https://github.com/ggml-org/llama.cpp/issues/28158) |
| Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) | Yes, `qwen4exp`: [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (merged 2026-08-27) | Yes, `Qwen4ExpVisionModel` in the same PR | Qwen3-Coder parser (same template markers, [template](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/chat_template.jinja)) | **No MTP in master.** [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) is an open draft | Vision and general hallucination vs 27B [#27886](https://github.com/ggml-org/llama.cpp/issues/27886) (open); `-sm tensor` disabled for this arch (llama-arch.cpp), re-enable in open [#28569](https://github.com/ggml-org/llama.cpp/pull/28569), CUDA abort [#27964](https://github.com/ggml-org/llama.cpp/issues/27964); non-deterministic QSA top-k on CUDA [#28497](https://github.com/ggml-org/llama.cpp/issues/28497) |
| Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`) | Yes, `qwen35moe` ([#19435](https://github.com/ggml-org/llama.cpp/pull/19435)) | Yes, qwen3vl | Qwen3-Coder parser | MTP yes ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673)); EAGLE3 [#24593](https://github.com/ggml-org/llama.cpp/pull/24593) | `tool_choice` required/named not enforced with thinking off [#27767](https://github.com/ggml-org/llama.cpp/issues/27767); MTP emits `////` in long sessions (27B) [#23577](https://github.com/ggml-org/llama.cpp/issues/23577); `--fit-target` regression [#27171](https://github.com/ggml-org/llama.cpp/issues/27171) |
| MiniMax-M2.7 (`MiniMaxM2ForCausalLM`) | Yes, `minimax-m2`: [#16831](https://github.com/ggml-org/llama.cpp/pull/16831) (2025-10-31) | No (text-only model) | Autoparser: the `<minimax:tool_call><invoke>` XML is not in the specialized list (original XML parser [#16932](https://github.com/ggml-org/llama.cpp/pull/16932)) | No MTP; EAGLE3 fix [#25604](https://github.com/ggml-org/llama.cpp/pull/25604) | A specialized M2 parser PR was closed unmerged [#22106](https://github.com/ggml-org/llama.cpp/pull/22106); no `-sm tensor` |
| MiniMax-M3 (`MiniMaxM3SparseForConditionalGeneration`) | Yes, `minimax-m3`: [#24908](https://github.com/ggml-org/llama.cpp/pull/24908) (2026-07-26) | Yes, [#25113](https://github.com/ggml-org/llama.cpp/pull/25113) | Specialized M3 parser [#26210](https://github.com/ggml-org/llama.cpp/pull/26210) | No MTP; EAGLE3 open [#24925](https://github.com/ggml-org/llama.cpp/pull/24925) | Quantized KV added [#26180](https://github.com/ggml-org/llama.cpp/pull/26180); unsloth repo has **no mmproj**, but bartowski and AesSedai do (Task 3) |
| GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) | **No.** Three competing open PRs: [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) (text only), [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) (+vision; needs `NVIDIA_TF32_OVERRIDE=0 -fa off`), [#27773](https://github.com/ggml-org/llama.cpp/pull/27773) (+vision) | Only in PRs #27754 and #27773 | Would use the autoparser (`<arg_key>` GLM style, [template](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja)) | MTP open draft [#27917](https://github.com/ggml-org/llama.cpp/pull/27917) | CUDA SOFT_MAX failure on Turing [#28144](https://github.com/ggml-org/llama.cpp/issues/28144), illegal memory access on long prefill [#28282](https://github.com/ggml-org/llama.cpp/issues/28282), RPC [#28360](https://github.com/ggml-org/llama.cpp/issues/28360) (all against the PR branches) |
| GLM-5.3 (`GlmMoeDsaForCausalLM`) | Yes, `glm-dsa` (same arch as GLM-5.2): [#19460](https://github.com/ggml-org/llama.cpp/pull/19460), indexer [#25407](https://github.com/ggml-org/llama.cpp/pull/25407) | No (text-only) | Autoparser (`<arg_key>`) | MTP yes: [#25980](https://github.com/ggml-org/llama.cpp/pull/25980); config has `num_nextn_predict_layers: 1` ([config](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json)) | GLM_DSA CUDA dense-MLA output corruption [#26027](https://github.com/ggml-org/llama.cpp/issues/26027) (open); multi-node RPC crash [#26583](https://github.com/ggml-org/llama.cpp/issues/26583) |
| DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`) | **No.** Open draft converter [#28696](https://github.com/ggml-org/llama.cpp/pull/28696); V4 (`deepseek4`) is supported | V4 vision works ([#28154](https://github.com/ggml-org/llama.cpp/pull/28154)); V4.1 not supported | Card: "does not include a Jinja-format chat template" ([card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)) | n/a | No unsloth or bartowski GGUF |
| Kimi-K2.7-Code (`KimiK25ForConditionalGeneration`) | Yes (DeepSeek2 path): [#19170](https://github.com/ggml-org/llama.cpp/pull/19170) (2026-02-11) | Yes, `kimik25` projector (conversion/kimivl.py) | Specialized Kimi K2 parser | No MTP | Past grammar bug with 22 tools [#24658](https://github.com/ggml-org/llama.cpp/issues/24658) (closed) |
| Kimi-K3 (`KimiK3ForConditionalGeneration`) | Yes, `kimi-k3`: [#26185](https://github.com/ggml-org/llama.cpp/pull/26185) (2026-08-15) | **No:** `kimik3` projector rejected [#28264](https://github.com/ggml-org/llama.cpp/issues/28264) (open) | Specialized Kimi K3 parser | Spec-decoding rollback [#28466](https://github.com/ggml-org/llama.cpp/pull/28466); no MTP | RPC crash on Apple [#28126](https://github.com/ggml-org/llama.cpp/issues/28126) |
| Laguna-S-2.1 / XS-2.1 / M.1 (`LagunaForCausalLM`) | Yes, `laguna`: [#25165](https://github.com/ggml-org/llama.cpp/pull/25165) (**merged 2026-07-22 01:54Z**, lead confirmed); S-2.1 fixes [#26232](https://github.com/ggml-org/llama.cpp/pull/26232), [#26233](https://github.com/ggml-org/llama.cpp/pull/26233) | No (text-only) | Autoparser (`<arg_key>`, [S template](https://huggingface.co/poolside/Laguna-S-2.1/raw/main/chat_template.jinja)) | **DFlash is not usable upstream for Laguna-S.** `draft-dflash` exists in master, but the Laguna drafter fails to load (`expected 76, got 69`) on master b10665. A minimal patch is posted in [#26669](https://github.com/ggml-org/llama.cpp/issues/26669) and not merged. The card says full DFlash support is in the [poolsideai `laguna` branch](https://github.com/poolsideai/llama.cpp/tree/laguna) (lead confirmed). No MTP | **CUDA NaN logits for some k-quant mixes of Laguna-S** [#27899](https://github.com/ggml-org/llama.cpp/issues/27899) (open, relevant to us); `-sm tensor` beyond 4 GPUs in open PR [#24554](https://github.com/ggml-org/llama.cpp/pull/24554) |
| Mistral-Medium-3.5-128B (`Mistral3ForConditionalGeneration`, dense) | Yes (mistral3 + llava/pixtral); conversion fix [#24268](https://github.com/ggml-org/llama.cpp/pull/24268) | Yes (pixtral) | Specialized Ministral/Magistral-3 parser (`[TOOL_CALLS][ARGS]`, no `[CALL_ID]`) | EAGLE drafter exists on HF ([Mistral-Medium-3.5-128B-EAGLE](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B-EAGLE)); llama.cpp compatibility not verified | Card warns that GGUFs made before a config fix degrade at long context ([card](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B)) |

## Task 2: Quality numbers

Vendor numbers (V) come from each model's HF card at `https://huggingface.co/<repo>`. The MiniMax and GLM-Flash tables are images: [M3](https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/figures/benchmark.jpeg), [M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7/resolve/main/figures/benchmark_overview.png), [GLM-5.3-Flash](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_53.png).

Independent numbers (I) come from three places:
- **AA**: Artificial Analysis model pages at `https://artificialanalysis.ai/models/<slug>`. Values were extracted from the page JSON: Intelligence Index, Terminal-Bench 2.1, Terminal-Bench Hard, tau2, MMMU-Pro.
- **DSWE**: the DeepSWE v1.1 leaderboard (mini-swe-agent) at https://deepswe.datacurve.ai/.
- **TOOL**: Toolathlon at https://toolathlon.xyz/.

Harnesses and benchmark versions differ, so compare numbers only within a column and source.

| Model | Active / total | SWE-bench V / Pro (V) | Terminal-Bench (V) | Other agentic (V) | LCB v6 (V) | **AA-I Index / TB2.1 / TB-Hard / tau2 (I)** | **DeepSWE / Toolathlon (I)** | Vision |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 27.8B dense | – / 61.7 | TB2.1 73.0 | DeepSWE 42.2 | 90.3 | 33.9 / 79.8 / – / – ([AA](https://artificialanalysis.ai/models/qwen3-8-27b)) | not found | V: OSWorld-Verified 84.3, MathVision 90.0; I: AA MMMU-Pro 76.3 |
| Qwen3.8-Flash-Next | 6B / 180B (125B + 51B n-gram emb + 4B MTP, [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) | – / 62.5; Multilingual 81.0 | – | DeepSWE 58.7, Toolathlon-V 73.5 | 91.9 | 39.9 / **86.1** / – / – ([AA](https://artificialanalysis.ai/models/qwen3-8-flash-next)) | not found | V: AndroidWorld 84.5, MathVision 90.6; I: AA MMMU-Pro 79.8 |
| Qwen3.6-35B-A3B | 3B / 36B | 73.4 / 49.5 | TB2.0 51.5 | TAU3 67.2 | 80.4 | 18.8 / 44.9 / 34.8 / 95.3 ([AA](https://artificialanalysis.ai/models/qwen3-6-35b-a3b)) | not found | V: MMMU 81.7, MMMU-Pro 75.3; I: AA MMMU-Pro 75.0 |
| MiniMax-M2.7 | 10B (AA) / 229B | 79.9 (M3 table) / 56.2 | TB2 57.0; TB2.1 51.1 (M3 table) | Toolathlon 46.3 | not found | 23.2 / 55.4 / 39.4 / 84.8 ([AA](https://artificialanalysis.ai/models/minimax-m2-7)) | not found | text-only |
| MiniMax-M3 | ~23B / 427B | 80.5 / 59.0 | TB2.1 66.0 | MCP Atlas 74.2, OSWorld-V 75.2 | not found | 29.6 / 65.2 / 42.4 / 88.9 ([AA](https://artificialanalysis.ai/models/minimax-m3)) | not found | V: MMMU-Pro 78.1; I: AA MMMU-Pro 78.6 |
| GLM-5.3-Flash | 18B / 321B | not found | TB2.1 84.3 | DeepSWE 63.4 | not found | **41.9 / 84.3** / – / – ([AA](https://artificialanalysis.ai/models/glm-5-3-flash)) | DSWE 63±4; TOOL 78.4±1.9 | V: BabyVision reported in footnotes only; I: not found |
| GLM-5.3 | 40B (AA) / 753B | not found | TB2.1 88.2, TB3.0 28.3 | DeepSWE 66.9, Toolathlon-V 73.0 | not found | 44.9 / 83.9 / – / – ([AA](https://artificialanalysis.ai/models/glm-5-3)) | DSWE 69±3 | text-only |
| DeepSeek-V4.1-Flash | 8B/16B / 763B | not found | TB2.1 90.6, TB3.0 30.0 | DeepSWE 74.2 | not found | 39.5 / – ([AA](https://artificialanalysis.ai/models/deepseek-v4-1-flash)) | not found | V: BabyVision w/ tools 89.6; I: AA MMMU-Pro 77.0 |
| Kimi-K2.7-Code | 32B / 1T | not found | not found | MCP Atlas 76.0, MCPMark-V 81.1 | not found | 26.3 / 67.4 / 44.7 / 90.1 ([AA](https://artificialanalysis.ai/models/kimi-k2-7-code)) | not found | V: not found |
| Kimi-K3 | 104B / 2.8T | not found | TB2.1 88.3 | DeepSWE 67.5, Toolathlon-V 76.5 | not found | 43.8 / 85.0 / – / – ([AA](https://artificialanalysis.ai/models/kimi-k3)) | DSWE 69±5; TOOL 76.5±1.9 | V: MMMU-Pro 81.6; I: AA MMMU-Pro 80.5 |
| Laguna-S-2.1 | ~8B / 117.6B | – / **59.4**; Multilingual 78.5 | **TB2.1 70.2** | DeepSWE 40.4, Toolathlon-V 49.7 | not found | **not found** (AA page 404) | **not found** | text-only |
| Laguna-XS-2.1 | 3B / 33.4B | 70.9 / 47.6 | TB2.0 37.5 | – | not found | not found (AA 404) | not found | text-only |
| Laguna-M.1 | 23B / 225.8B | 74.6 / 49.2 | TB2.0 45.8 | – | not found | not found (AA 404) | not found | text-only |
| Mistral-Medium-3.5 | 128B dense | 77.6 / – | – | tau3-Telecom 91.4 | not found | 14.9 / 50.6 / 33.3 / 94.2 ([AA](https://artificialanalysis.ai/models/mistral-medium-3-5)) | not found | I: AA MMMU-Pro 64.9 |

Notes on the coordinator's Laguna leads:
- The Laguna-S card numbers of TB2.1 70.2 and SWE-Pro 59.4 are confirmed ([card](https://huggingface.co/poolside/Laguna-S-2.1)). They are vendor numbers from Poolside's own `pool` harness. The card's asterisk marks only the comparison models' scores as third-party.
- I found no independent Laguna scores. Laguna is absent from Artificial Analysis (model pages return 404), the DeepSWE leaderboard, and Toolathlon, and I could not read any Laguna entries from the Terminal-Bench, Scale SWE-Bench Pro, swebench.com, or Aider leaderboard pages.
- Aider polyglot and BFCL: none of these 2026 models appear in the Aider leaderboard HTML (https://aider.chat/docs/leaderboards/), and I found no BFCL numbers.
- Laguna-XS-2.1 vs Qwen3.8-27B on a single GPU: the lead is **refuted on quality**. Qwen3.8-27B reports SWE-Pro 61.7 vs 47.6 (both vendor), and Qwen3.8-27B has an independent AA TB2.1 of 79.8, while XS has no independent score. Laguna's own card also shows XS-2.1 below Qwen3.6-35B-A3B on all four columns (SWE-V 70.9 vs 73.4, TB2.0 37.5 vs 51.5). XS wins only on decode speed (3B active) and lacks vision.

## Task 3: GGUF fit on our hardware

Sizes are summed GiB from the HF tree API, `https://huggingface.co/api/models/<repo>/tree/main?recursive=true`, with split shards added together. Budgets are (a) ≤154, (b) ≤110, (c) ≤18.3 GiB, as measured above. The mmproj file is added for vision use.

| Model | GGUF repo | Smallest / largest | (a) 8 GPUs, ≤154 GiB | (b) 6 GPUs, ≤110 GiB | (c) 1 GPU, ≤18.3 GiB | Active params |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 5.8 / 50.9 (BF16); mmproj F16 0.86; MTP 1.3 | BF16 50.9 | BF16 50.9 | **UD-Q4_K_XL 16.4 + mmproj 0.86 = 17.3**, or text-only UD-Q5_K_S 17.4 | 27.8B |
| Qwen3.8-Flash-Next | [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 67.6 / 329.7; mmproj 0.84; MTP 22.9 | **UD-Q5_K_XL 147.4 (+0.84)**; Q6_K_XL 157.5 does not fit | UD-Q4_K_XL 103.7 (+0.84) | no (needs CPU MoE offload) | 6B |
| Qwen3.6-35B-A3B | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 9.4 / 64.6; mmproj 0.84 | BF16 64.6 | BF16 64.6 | **UD-IQ4_NL 16.8 + 0.84 = 17.6** | 3B |
| MiniMax-M2.7 | [unsloth](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF) | 56.5 / 426.1 | UD-Q5_K_S 148.1 | UD-IQ4_NL 103.1 | no | 10B |
| MiniMax-M3 | [unsloth](https://huggingface.co/unsloth/MiniMax-M3-GGUF) (mmproj only in [AesSedai](https://huggingface.co/AesSedai/MiniMax-M3-GGUF): Q8_0 0.86, F16 1.61) | 119.6 / 793.5 | UD-IQ3_XXS 148.5 (+0.86) | no (IQ1_M is 119.6) | no | ~23B |
| GLM-5.3-Flash (PR-only) | [unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | 86.7 / 597.6; mmproj F16 1.05 | UD-IQ4_XS 146.1 (+1.05) | UD-Q2_K_XL 101.3 (IQ3_XXS 112.1 misses) | no | 18B |
| GLM-5.3 | [unsloth](https://huggingface.co/unsloth/GLM-5.3-GGUF) | 201.8 / 1404.4 | no | no | no | 40B |
| DeepSeek-V4.1-Flash | no unsloth or bartowski repo; [antirez](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) Q2 is 340.6 GiB for DwarfStar, not llama.cpp | – | no (and unsupported) | no | no | 8B/16B |
| Kimi-K2.7-Code | [unsloth](https://huggingface.co/unsloth/Kimi-K2.7-Code-GGUF) | 283.0 / 553.7 | no (needs about 130+ GiB in RAM) | no | no | 32B |
| Kimi-K3 | [unsloth](https://huggingface.co/unsloth/Kimi-K3-GGUF) | 434.3 / 1453.9 | no | no | no | 104B |
| Laguna-S-2.1 | [unsloth](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF), [poolside](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) (Q8_0 **119.9**, Q4_K_M **89.4**, DFlash 2.1; leads confirmed) | 31.4 / 219.0 | UD-Q8_K_XL 119.3 | UD-Q6_K_XL 99.7 | no (IQ1_S is 31.4) | ~8B |
| Laguna-XS-2.1 | [bartowski](https://huggingface.co/bartowski/Laguna-XS-2.1-GGUF), [poolside](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) (Q4_K_M 18.9) | 8.8 / 62.3 | BF16 62.3 | BF16 62.3 | IQ4_NL 17.9 or IQ4_XS 17.0 (Q4_K_M 19.1 misses) | 3B |
| Laguna-M.1 | no unsloth or bartowski repo; [sigargv](https://huggingface.co/sigargv/Laguna-M.1-GGUF) (third-party) | 119.9 / 420.7 | Q5_K_M 149.3 | no (Q4_K_S is 119.9) | no | 23B |
| Mistral-Medium-3.5 | [unsloth](https://huggingface.co/unsloth/Mistral-Medium-3.5-128B-GGUF) | 32.5 / 232.9; mmproj F16 **4.99** | Q8_0 123.7 + 5.0 | Q6_K 95.5 + 5.0 | no | 128B dense (slow decode) |

We have 1 TB of RAM, so the models marked "no" can still run in hybrid mode with MoE experts in RAM (`--n-cpu-moe`), but decode will be much slower. I did not measure that.

## Task 4: Recommendations

**(A) Main agentic coding model on 6–8 A5000s**
1. **Qwen3.8-Flash-Next.** Use UD-Q5_K_XL on 8 GPUs or UD-Q4_K_XL on 6.
   - It has the highest independent TB2.1 among models that fit: AA 86.1.
   - It uses only 6B active params and supports vision.
   - Risks: the arch is new in master, [#27886](https://github.com/ggml-org/llama.cpp/issues/27886) reports hallucination and vision problems, MTP has not merged, and `-sm tensor` is off. A/B it against #2 on our own tasks before committing.
2. **Qwen3.8-27B** at BF16 or Q8 as the stable choice.
   - It scores AA TB2.1 79.8 and vendor SWE-Pro 61.7.
   - Its arch (`qwen35`) has been mature since February, with MTP and DFlash2 drafters.
   - It fits on 2–3 GPUs, which frees the rest for parallel slots.
3. **GLM-5.3-Flash**, but only once one of #27752, #27754 or #27773 merges. Use UD-IQ4_XS on 8 GPUs.
   - It has the best independent overall profile that fits: AA Intelligence 41.9, TB2.1 84.3, DeepSWE 63, Toolathlon 78.4.
   - It is **not in master today**.

Next in line is Laguna-S-2.1 (UD-Q8_K_XL on 8 GPUs, Q6_K_XL on 6). Its SWE-Pro 59.4 and TB2.1 70.2 are vendor-only, it has no vision, upstream DFlash needs a patch, and the open CUDA NaN bug [#27899](https://github.com/ggml-org/llama.cpp/issues/27899) affects our backend. MiniMax-M3 only fits at IQ3 on 8 GPUs, with AA TB2.1 65.2. GLM-5.3, Kimi and DeepSeek-V4.1 do not fit in VRAM (and V4.1 is unsupported).

**(B) Single 24 GB GPU (22.5 GiB usable)**
1. **Qwen3.8-27B**: UD-Q4_K_XL plus mmproj (17.3 GiB) with MTP. AA TB2.1 is 79.8, far ahead of the others.
2. **Qwen3.6-35B-A3B**: UD-IQ4_NL plus mmproj (17.6 GiB). It is much faster with 3B active, but weaker: AA TB2.1 44.9, vendor SWE-V 73.4.
3. **Laguna-XS-2.1**: IQ4_NL (17.9 GiB). It is text-only and has only vendor numbers (SWE-V 70.9, SWE-Pro 47.6, TB2.0 37.5), which trail Qwen3.6-35B-A3B on Poolside's own table.

**(C) Vision**
1. **Qwen3.8-27B**: AA MMMU-Pro 76.3, vendor OSWorld-Verified 84.3, and vision works in master with no open vision bug.
2. **Qwen3.8-Flash-Next**: higher scores (AA MMMU-Pro 79.8, vendor AndroidWorld 84.5), but vision quality in llama.cpp is disputed in [#27886](https://github.com/ggml-org/llama.cpp/issues/27886).
3. **MiniMax-M3**: AA MMMU-Pro 78.6. Vision is merged ([#25113](https://github.com/ggml-org/llama.cpp/pull/25113)), but it only fits at UD-IQ3_XXS on 8 GPUs, with the mmproj from AesSedai or bartowski.
   - Alternative: Qwen3.6-35B-A3B (AA MMMU-Pro 75.0) if a single GPU must also do vision.
   - Kimi-K3 vision is not supported in master ([#28264](https://github.com/ggml-org/llama.cpp/issues/28264)).

## Frontier comparison (follow-up)

All **I** (independent) numbers come from Artificial Analysis model pages, `https://artificialanalysis.ai/models/<slug>`, extracted from each page's JSON, so open and closed models share one harness. **V** numbers come from the vendor's own HF card or blog. DeepSWE **I** numbers come from https://deepswe.datacurve.ai/ (mini-swe-agent) and Toolathlon **I** numbers from https://toolathlon.xyz/.

### 1. Agentic coding

| Model | AA Intelligence Index (I) | Terminal-Bench 2.1 (I) | DeepSWE v1.1 (I) | Toolathlon (I) | SWE-bench Pro / Verified (V) |
|---|---|---|---|---|---|
| **Qwen3.8-27B** (open, 27.8B) | 33.9 | **79.8** ([AA](https://artificialanalysis.ai/models/qwen3-8-27b)) | not on leaderboard | not on leaderboard | Pro 61.7, TB2.1 73.0 ([card](https://huggingface.co/Qwen/Qwen3.8-27B)); Toolathlon-V 67.1 |
| **Qwen3.8-Flash-Next** (open, 6B active) | 39.9 | **86.1** ([AA](https://artificialanalysis.ai/models/qwen3-8-flash-next)) | not on leaderboard | not on leaderboard | Pro 62.5, DeepSWE 58.7, Toolathlon-V 73.5 ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) |
| Claude Fable 5.1 (max) | **53.4** | **91.4** ([AA](https://artificialanalysis.ai/models/claude-fable-5-1)) | 70±3 (Fable 5) | – | – |
| Claude Opus 5 (max) | 50.7 | 89.1 ([AA](https://artificialanalysis.ai/models/claude-opus-5)) | **74±4** | – | – |
| GPT-6 Astra (max) | 52.8 | 88.4 ([AA](https://artificialanalysis.ai/models/gpt-6-astra)) | **74±3** | – | – |
| GPT-5.6 Sol (max) | 47.1 | 88.0 ([AA](https://artificialanalysis.ai/models/gpt-5-6-sol)) | 73±3 | – | – |
| GPT-5.6 Terra (max) | 42.3 | 88.0 ([AA](https://artificialanalysis.ai/models/gpt-5-6-terra)) | – | – | – |
| Gemini 3.8 Flash (high) | 41.2 | 87.6 ([AA](https://artificialanalysis.ai/models/gemini-3-8-flash)) | **74±1** | – | – |
| Grok 4.6 (high) | 44.4 | 88.4 ([AA](https://artificialanalysis.ai/models/grok-4-6)) | – | – | – |
| Claude Opus 4.8 (max) | – | – | 59±2 | 76.2±3.4 | – |
| GPT-5.5 (xhigh) | – | – | 67±6 | 73.5±1.2 | – |
| *Reference: GLM-5.3 / Kimi K3 (open)* | 44.9 / 43.8 | 83.9 / 85.0 | 69±3 / 69±5 | – / 76.5±1.9 | – |

Neither Qwen3.8 model appears on the DeepSWE or Toolathlon leaderboards, so those columns are vendor-only for them. Artificial Analysis publishes no SWE-bench Pro or Verified column, so that column is vendor-only for everyone, and the Qwen card's SWE-Pro comparison (61.7 vs Opus 4.6 Max at 53.4) uses Qwen's own corrected task set and its own Claude Code harness.

### 2. Vision

Both Qwen3.8 models take image input, and Artificial Analysis marks both as accepting **video** as well, which most closed frontier models do not.

| Model | Image / video input (I) | MMMU-Pro (I) | Vendor vision detail (V) |
|---|---|---|---|
| **Qwen3.8-27B** | image yes, video yes | **76.3** | OSWorld-Verified 84.3, AndroidWorld 81.9, OmniDocBench 1.5 91.1, Vision2Web 62.9, MathVision 90.0 ([card](https://huggingface.co/Qwen/Qwen3.8-27B)) |
| **Qwen3.8-Flash-Next** | image yes, video yes | **79.8** | AndroidWorld 84.5, OSWorld 2.0 binary 19.4 / partial 52.3, Vision2Web 64.0, LVBench 76.6 ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) |
| GPT-6 Astra (max) | image yes, video no | **86.9** | – |
| Gemini 3.8 Flash (high) | image, video and speech | 85.6 | – |
| Claude Opus 5 (max) | image yes, video no | 84.7 | – |
| GPT-5.6 Sol (max) | image yes, video no | 83.4 | – |
| Kimi K3 (open) | image yes, video no | 80.5 | – |
| GLM-5.3 (open) | **no image input** | – | – |

What these numbers do **not** cover:
- Artificial Analysis reports only MMMU-Pro for vision. It has **no OCR, screenshot or GUI-grounding score**, so the OSWorld, AndroidWorld, Vision2Web and OmniDocBench numbers above are vendor-only and were measured with the vendor's own scaffold, not llama.cpp.
- I found **no independent GUI-grounding or ScreenSpot number** for any of these models.
- In llama.cpp specifically, Qwen3.8-Flash-Next vision quality is disputed: [#27886](https://github.com/ggml-org/llama.cpp/issues/27886) reports missed image regions and document errors versus Qwen3.8-27B on the same build, with one report of a resolution or format dependence. Treat the vendor vision scores as an upper bound on what our stack will deliver.
- Video input via mmproj in llama.cpp is not something I verified for these models; the vendor video scores (LVBench, VideoMME) come from their own stacks.

### 3. The gap in one line

On agentic terminal coding measured by one harness, Qwen3.8-Flash-Next reaches **94%** of the frontier leader (Terminal-Bench 2.1: 86.1 vs Claude Fable 5.1's 91.4) and Qwen3.8-27B reaches **87%** (79.8), but on the broader Intelligence Index the same models reach only **75%** and **64%** (39.9 and 33.9 vs 53.4) — the coding-specific gap is much smaller than the general-capability gap, and harnesses, reasoning effort and scaffolds differ between the vendor and independent numbers.

### 4. Long context at 128k+

- **Context windows (I, Artificial Analysis):** Qwen3.8-27B and Flash-Next are listed at **256k**, while Claude Fable 5.1, Claude Opus 5, GPT-6 Astra, GPT-5.6 Sol/Terra and Gemini 3.8 Flash are all listed at **1M** and Grok 4.6 at 500k. Both Qwen cards say 262,144 native and up to about 1M only with YaRN RoPE scaling, which they advise enabling just for long-context work.
- **AA-LCR long-context reasoning (I):** Qwen3.8-27B **82.0**, Flash-Next **79.7**, versus Claude Fable 5.1 85.3, GPT-5.6 Sol 84.0, Gemini 3.8 Flash 81.3, Grok 4.6 80.3, Claude Opus 5 79.3, and Kimi K3 highest at 88.7. So Qwen3.8-27B is at about **96%** of the frontier leader on this one long-context metric — much closer than the general index gap.
- **Practical llama.cpp caveats at long context:** qwen4exp (Flash-Next) on Metal emits one token then EOS at long context ([#28805](https://github.com/ggml-org/llama.cpp/issues/28805), Metal-only, so likely not our CUDA path but worth watching); GLM-5.3-Flash hits a CUDA illegal memory access on long prefill at `-ub 2048` ([#28282](https://github.com/ggml-org/llama.cpp/issues/28282)); and the Qwen3.6 MTP `////` repetition in long sessions ([#23577](https://github.com/ggml-org/llama.cpp/issues/23577)) is a reason to test MTP at session length rather than trusting short benchmarks.
- Qwen's card also notes that thinking blocks are preserved across turns by default, which raises KV use in long agent sessions but improves cache reuse.

## VQA (follow-up)

### What exists and what does not

Of the benchmarks asked about, **VQAv2, MMStar, ChartQA, InfoVQA, TextVQA, OCRBench and ScreenSpot / ScreenSpot-Pro appear on none of these four model cards** — not found. DocVQA appears only on the DeepSeek-V4.1-Flash card (95.6, LLM-judge, 4-shot, base model, [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). Artificial Analysis publishes **only MMMU-Pro** for vision (its per-model JSON has `mmmuPro` and no other vision field), so **every number in the table below is vendor-reported (V)** and I found no independent VQA-style numbers for these models.

The cards use a house style of RealWorldQA, MMBench, MathVista/MathVision, AI2D, CharXiv, OmniDocBench, CC-OCR and OSWorld instead of the older VQA set.

### VQA and perception scores (all vendor-reported)

Each card carries its own frontier column, which is the only place one source has both. Frontier reference in each row is in parentheses.

| Benchmark | Qwen3.8-27B ([card](https://huggingface.co/Qwen/Qwen3.8-27B)) | Qwen3.8-Flash-Next ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)) | Qwen3.6-35B-A3B ([card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | MiniMax-M3 ([figure](https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/figures/benchmark.jpeg)) |
|---|---|---|---|---|
| RealWorldQA | 85.9 (Opus 4.6 Max 73.9) | **88.5** (Opus 4.6 Max 73.9) | 85.3 (Sonnet 4.5 70.3) | not found |
| MMBench EN-DEV v1.1 | not found | not found | 92.8 (Sonnet 4.5 88.3) | not found |
| MathVista (mini) | not found | not found | 86.4 (Sonnet 4.5 79.8) | not found |
| MathVision | 90.0 / 94.6 with CI (Opus 4.6 Max 65.5) | **90.6 / 95.7** (Opus 4.6 Max 65.5) | not found | not found |
| AI2D (test) | not found | not found | 92.7 (Sonnet 4.5 87.0) | not found |
| OmniDocBench 1.5 (doc OCR) | 91.1 (Opus 4.6 Max 86.6) | not found | 89.9 (Sonnet 4.5 85.8) | **91.6** (Opus 4.7 89.3, GPT-5.5 87.5, Gemini 3.1 Pro 88.1) |
| CC-OCR | not found | not found | 81.9 (Sonnet 4.5 68.1) | not found |
| CharXiv (RQ, chart reasoning) | 83.7 / 90.2 (Opus 4.6 Max 66.0) | 84.6 / 90.6 | 78.0 (Sonnet 4.5 67.2) | not found |
| MMMU / MMMU-Pro | MMMU-Pro 76.3 (**I**, [AA](https://artificialanalysis.ai/models/qwen3-8-27b)) | MMMU-Pro 79.8 (**I**, [AA](https://artificialanalysis.ai/models/qwen3-8-flash-next)) | MMMU 81.7, MMMU-Pro 75.3 (Sonnet 4.5 79.6 / 68.4) | MMMU-Pro 78.1 (Opus 4.7 77.0, GPT-5.5 81.2, Gemini 3.1 Pro 80.5) |
| HallusionBench | not found | not found | 69.8 (Sonnet 4.5 59.9) | not found |
| SimpleVQA | not found | not found | 58.9 (Sonnet 4.5 57.6) | not found |
| RefCOCO (grounding) | not found | not found | 92.0 avg; ODInW13 50.8 | not found |
| ZeroBench (sub) | not found | not found | 34.4 (Sonnet 4.5 26.3) | not found |
| **OSWorld** | OSWorld-Verified **84.3** (Opus 4.6 Max 72.7) | OSWorld 2.0 binary 19.4 / partial 52.3 | not found | OSWorld-Verified 75.2 (Opus 4.7 82.8, GPT-5.5 78.7, Gemini 3.1 Pro 76.2) |
| Other GUI / web | WebArena-Verified 64.8; AndroidWorld 81.9 (Opus 4.6 Max 62.0); Vision2Web 62.9; SWE-MM 38.6 (Opus 4.6 Max 27.1) | AndroidWorld **84.5**; Vision2Web 64.0; ClawEval-MM 64.4 pass@3 | EmbSpatialBench 84.3 (Sonnet 4.5 71.8) | not found |
| Video | not found | LVBench 76.6; ERQA 72.3 | VideoMME w/ sub 86.6; VideoMMMU 83.7; MLVU 86.2; LVBench 71.4 | Video-MMMU 84.6; VideoMME w/ sub 85.4 |

Independent MMMU-Pro reference points for the frontier, from the same Artificial Analysis harness: GPT-6 Astra 86.9, Gemini 3.8 Flash 85.6, Claude Opus 5 84.7, GPT-5.6 Sol 83.4. So the Qwen models' 76.3 and 79.8 sit 6–11 points below the frontier on the one vision benchmark with a shared harness, while their vendor-reported OSWorld and AndroidWorld numbers claim a lead over Claude. Those agentic-GUI claims use Qwen's own scaffold and have no independent confirmation.

### llama.cpp practicalities

**1. Does the unsloth mmproj keep the original resolution and tiling? Partly — the default cap is about 4x lower than the model allows.**
- I parsed the GGUF headers of the unsloth mmproj files directly. `unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf`, `Qwen3.8-Flash-Next` and `Qwen3.6-35B-A3B` all declare `clip.projector_type = qwen3vl_merger`, `clip.vision.patch_size = 16`, `clip.vision.spatial_merge_size = 2`, and **none of them set `clip.vision.image_min_pixels` or `image_max_pixels`**.
- With those keys absent, llama.cpp falls back to the hard-coded Qwen-VL limits in [clip.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip.cpp): `hparams.set_limit_image_tokens(8, 4096)` for `PROJECTOR_TYPE_QWEN3VL`, citing the Qwen2.5-VL preprocessor config. At patch 16 with merge 2, one image token covers 32x32 pixels, so the default ceiling is **4096 tokens, about 4.2 Mpx**.
- Qwen's own `preprocessor_config.json` for all three models allows `size.longest_edge = 16777216` pixels, i.e. about **16.8 Mpx** ([Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/preprocessor_config.json)). So llama.cpp downsamples large images roughly 4x more aggressively than the reference stack by default. This is a resolution cap, not a different tiling scheme: Qwen3-VL uses native dynamic resolution rather than fixed tiles, and llama.cpp implements the same merger.
- llama.cpp itself warns about the low end: "Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks... try adding `--image-min-tokens 1024`" (clip.cpp). **Practical advice: pass `--image-min-tokens 1024` and raise `--image-max-tokens` if documents or screenshots look degraded** ([flags in common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)).
- docs/multimodal.md says nothing about resolution, tiling or quality — not found there.

**2. Open Qwen3.8 image issues beyond #27886**
- [#27419](https://github.com/ggml-org/llama.cpp/issues/27419) (open): "upper image content is lost while Vulkan/CPU are correct", reported against exactly our file pair, `unsloth/Qwen3.8-27B-GGUF` Q4_K_XL plus its mmproj, on HIP/gfx1151. It is a backend bug rather than a preprocessing bug, but it is the second independent report of Qwen3.8 losing part of an image.
- [#28608](https://github.com/ggml-org/llama.cpp/issues/28608) (open): the CLIP vision encoder crashes on ROCm with large images in `flash_attn_tile`.
- [#28954](https://github.com/ggml-org/llama.cpp/issues/28954) (open): images above about 1.2 Mpx assert with Gemma4 — the reporter explicitly lists **Qwen3.8-27B as unaffected**.
- I found **no open issue on Qwen3.8 resizing or aspect-ratio handling specifically** — not found. Searches for preprocessing, resize and aspect-ratio issues returned only the MiniCPM-V mmproj overflow [#27166](https://github.com/ggml-org/llama.cpp/issues/27166) and the Qwen2.5-VL image-dimension issue [#17534](https://github.com/ggml-org/llama.cpp/issues/17534).

**3. Multi-image and long image sequences in llama-server**
- [#27931](https://github.com/ggml-org/llama.cpp/issues/27931) (open) matters most for us: llama-server crashes with a stack overflow or access violation on **hybrid recurrent models (`qwen3_5_moe`) with mmproj when alternating text and image turns** — that is the Qwen3.6-35B-A3B family, in exactly the multi-turn, multi-image pattern an agent produces.
- [#27408](https://github.com/ggml-org/llama.cpp/issues/27408) (open): with `--spec-type draft-dflash` plus an mmproj, **every image request stalls for about 500 s and returns HTTP 500**, because image chunks leave positional holes in the draft KV cache. The reporter's patch stops the crash but yields no speculative benefit on images. **Speculative decoding and vision do not currently combine.**
- [#19466](https://github.com/ggml-org/llama.cpp/issues/19466) (open): KV-cache save and restore via `/slots` does not work for vision-enabled models, so long image sessions cannot be checkpointed.
- I found no open report of a per-request limit on the number of images; the older multi-image requests [#7364](https://github.com/ggml-org/llama.cpp/issues/7364) and [#14530](https://github.com/ggml-org/llama.cpp/issues/14530) are closed, and batching several images in one request is still not implemented as of those threads.
