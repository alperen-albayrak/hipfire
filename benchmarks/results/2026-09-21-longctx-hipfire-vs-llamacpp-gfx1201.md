# Long-context: hipfire vs llama.cpp vs radiance/vLLM on one R9700 (gfx1201), 2026-09-21

**One-line result: at 124K context llama.cpp is 7.5x faster at cold prefill and
4.0x faster at decode than hipfire, on the same GPU, in the same hour — while
carrying a 12% larger model and reading 46% more KV.** The same 124K prompt
that costs hipfire 20.5 minutes to first token costs llama.cpp 2.8.

hipfire is *faster* at short context. It collapses at long context; llama.cpp
degrades gently. That difference is the finding.

## Three-engine summary (added after the radiance run)

Same GPU, same day, ~128K-token wikitext prompts, DFlash2 where available:

| | hipfire | llama.cpp | **radiance/vLLM MXFP4** |
|---|---:|---:|---:|
| cold prefill @~128K | 101.2 t/s | 760.1 t/s | **~2,280 t/s** |
| decode @~128K | 7.8 (DFlash2) | 15.1 (no spec) | **~55 (DFlash2)** |
| wall: 128K prefill + 256 tok | ~1,259 s | ~185 s | **60.5 s** |

radiance is **22x hipfire / 3x llama.cpp on prefill**, **7x hipfire / 3.6x
llama.cpp on decode**. It also beats the akougkas.io llama.cpp production preset
(47.5 t/s at 131k) while using DFlash2 rather than MTP.

### radiance method and caveats

`codeberg.org/ggz14/radiance-vllm-mxfp4`, image `stilldeadcode/vllm-radiance:0.9.3`,
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (~19 GiB) rewritten to mtp-fp8 by its setup,
TP=1, `MAXLEN=131072`, `GPU_UTIL=0.98`, `kv_cache_dtype=fp8`, R4D attention
backend, `SPEC_METHOD=dflash SPEC=7`.

Decomposition avoided the prefix cache, which proved unreliable here (a repeat
of the same prompt hit once, then did not — the KV cache holds only 131,437
tokens, so one 128K request fills it):

| prompt tokens | generated | wall |
|---:|---:|---:|
| 124,742 | 16 | 55.0 s |
| 127,770 | 256 | 60.4 s |
| 128,265 | 256 | 60.9 s / 61.0 s |

The 16-token run is ~99% prefill => 124,742 / 54.6 s = ~2,280 t/s. Applying that
to the 256-token runs leaves ~4.6 s for 256 tokens => ~56 t/s decode. The single
prefix-cache hit that did occur (4.8 s for 256 tokens = 53 t/s) agrees from an
independent route. Three fresh prompts reproduced the cold wall within 1%
(60.4 / 60.9 / 61.0).

**`SPEC_METHOD=mtp` does not work on this stack.** Engine init fails with a
Dynamo assertion in the MTP head's `fc` layer under Quark fp8
(`input_quant_fp8.py:190`, `assert (scale is not None) == self.static`). Their
README documents `dflash` as the default and flags the MTP-native checkpoint as
not load-tested. So these numbers may have headroom.

**Only one full-context stream fits**: `GPU KV cache size: 131,437 tokens`,
`Maximum concurrency for 131,072 tokens per request: 1.00x`, 31.2 of 31.9 GiB
resident. Two concurrent 128K lanes need a shorter per-request context.

### 250K does not fit on radiance — tested 2026-09-21

`MAXLEN=262144` fails at engine init:

> To serve at least one request with the model's max seq len (262144), 8.82 GiB
> KV cache is needed, which is larger than the available KV cache memory
> (**4.84 GiB**)

`MAXLEN=143360` also fails (needs 5.2 GiB). `131072` works. **The practical
ceiling is ~133K.**

The binding constraint is the checkpoint, not the engine:

| | GiB |
|---|---:|
| MXFP4 checkpoint | 19 |
| DFlash2 drafter | 2 |
| activations + CUDA graphs | ~5 |
| **KV remaining** | **4.84** |
| KV needed @262,144 (fp8, 33.6 KB/token) | **8.82** |

ParoQuant does not help: `z-lab/Qwen3.8-27B-PARO` is "4.25 bits/weight, the same
as MXFP4". Dropping the drafter frees ~2 GiB (~203K) but costs most of the
decode advantage.

### The three-way trade, complete

| engine | prefill / decode @128K | context ceiling |
|---|---|---:|
| **radiance/vLLM** | **2,280 / ~55 t/s** | ~133K (VRAM) |
| **llama.cpp** | 760 / 15.1 t/s | ~130K (issue 27756) |
| **hipfire** | 101 / 7.8 t/s | **165K measured, 262K configured** |

**No engine on this card delivers both speed and 250K.** hipfire reaches further
for exactly the reason measured earlier: fwht3 K costs 23.8 KB/token against
radiance's 33.6 KB/token fp8 — 5.8 GiB vs 8.8 GiB at 250K. The fwht3 decision
buys context reach, and costs 22x the prefill time to use it.

**This revises the earlier "vLLM is out" conclusion in this document.** Upstream
vLLM builds for gfx1201 but ships CDNA-only AITER kernels; this fork supplies
the RDNA4 work upstream lacks, and the result is the fastest of the three by a
wide margin.

## Numbers

Both engines, cachy-01, AMD Radeon AI PRO R9700 (gfx1201, 32 GB), ROCm 7.2.4.
llama.cpp pinned to device 0 (`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`)
so the 760M iGPU could take no layers — re-running pinned changed nothing
(563.51 vs 563.49, 15.49 vs 15.50), confirming it never contributed.

| | llama.cpp | hipfire | ratio |
|---|---:|---:|---|
| **cold prefill @124K** | **760.1 tok/s** | 101.2 tok/s | **7.5x** |
| **wall clock, same 124K prompt** | **168 s** (incl. model load) | 1,229 s | **7.3x** |
| prefill @124K, marginal | **563.5 tok/s** | 43.6 tok/s (@~145K) | ~13x |
| decode @124K, unspeculated | **15.49 tok/s** | ~3.84 tok/s | **4.0x** |
| decode @124K, hipfire *with* DFlash2 | 15.49 | 7.8 tok/s | 2.0x |
| prefill, short ctx | 1187.3 (pp2048) | 920 (4K) | 1.3x |
| decode, short ctx, unspeculated | 27.61 | ~42.4 | **hipfire 1.5x** |

Degradation from short context to 124K:

| | llama.cpp | hipfire |
|---|---:|---:|
| prefill | **2.1x** | **9.1x** |
| decode | **1.8x** | **11x** |

### Memory efficiency at 124K decode

| | bytes/step | ms/step | achieved |
|---|---:|---:|---:|
| llama.cpp | 16.34 GiB weights + ~4.3 GB KV = 20.7 GiB | 64.5 | **~321 GB/s** (~50% roofline) |
| hipfire | 14.59 GiB + 2.96 GiB KV = 17.5 GiB | ~225 | **~78 GB/s** (~12% roofline) |

4.1x memory efficiency, matching the 4.0x throughput gap. The two independent
accountings agree.

## Method

- **hipfire**: production daemon, `qwen3.8-27b.mq4` (MQ4V2, 14.59 GiB), fwht3 K
  + Q8 V, DFlash2 on. 124,345-token prompt from the wikitext2 slice via
  `/v1/chat/completions`; `prefill_tok_s` / `decode_tok_s` / `tau` read from the
  response `timings`. tau was 2.03, so the unspeculated rate is 7.8 / 2.03.
- **llama.cpp cold prefill**: `llama-cli -f <the same 124K prompt file hipfire
  used> -n 16 -c 131072 -fa on -ctk q8_0 -ctv q8_0 -st --temp 0`, device 0 only.
  Reported `Prompt: 760.1 t/s | Generation: 15.1 t/s`, 168 s total wall
  including model load. The 15.1 t/s cross-checks llama-bench's 15.49 from a
  different code path.
- **llama.cpp**: build `ggml-org/llama.cpp` @ HEAD, HIP, `-DAMDGPU_TARGETS=gfx1201`.
  `llama-bench -fa on -ctk q8_0 -ctv q8_0 -p 2048 -n 64 -d 0,124000 -r 1`
  against `unsloth/Qwen3.8-27B-GGUF :: Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.34 GiB).
  No speculative decoding.

### Fairness notes — every one favours hipfire

- llama.cpp's model is **12% larger** (16.34 vs 14.59 GiB): more weight bytes
  per decode step.
- llama.cpp ran **q8_0 KV**, 46% more KV bytes than hipfire's fwht3 (2176 vs
  1488 B/pos/layer). fwht3 has no llama.cpp equivalent, and fwht3-vs-q8 measured
  within 0.7% on hipfire's own prefill, so this is not the explanation.
- llama.cpp has **no speculative decoding** here; hipfire's 7.8 tok/s already
  includes DFlash2.
- `pp2048 @ d124000` is *marginal* throughput at depth 124K. hipfire's 101.2 is
  the *average* over 0->124K; its marginal rate measured 43.6 tok/s over
  124K->165K. Comparing llama.cpp's marginal against hipfire's average
  understates the gap; marginal-vs-marginal is 563.5 vs 43.6.

llama.cpp wins by 4-7.5x despite all of it.

## Why — corroborated by kernel isolation

`bench_longctx_decode_attn` (same day, same box) isolated hipfire's decode
attention and found it costs 12.3 ms/layer at 124K, i.e. ~197 ms of a 260 ms
verify cycle. Two structural causes, measured:

1. **No KV reuse across the verify batch.** Time is exactly linear in batch
   (us/token flat at ~1500 from batch 8 to 128), so each of the 8 draft rows
   re-reads the whole KV. 8x redundancy.
2. **The GQA group IS already shared** — cutting query heads 6x (24 -> 4) cut
   time only 1.8x — so the head-folding llama.cpp does via `ncols2`
   (`fattn.cu:301`, `gqa_ratio = Q->ne[2] / K->ne[2]`) is *not* the missing
   piece. hipfire substantially has it.

Two further limits, both in-tree:

- `attention_q8_0_fa2_gqa_*_gfx1201` — the tuned kernel — is gated at
  `max_ctx_len <= 32768` (dispatch + launcher), so above 32K **neither** tier
  uses it. The bound is real but belongs only to the split path, whose
  partition hardcodes `T_MAX = 512` tiles
  (`attention_q8_0_fa2_gqa.gfx1201.hip:636`); the direct kernel is unbounded
  (`t0=0, t1=0x3fffffff`).
- FA2 also requires `batch_size` in `64..=512`, so a verify step of 8 can never
  reach it. Calling it directly at batch 8 was **slower** than the incumbent
  (2.595 vs 2.362 ms at 32K): its grid is `[ceil(batch/8), 4, 1]`, i.e. 4
  workgroups — reuse gained, occupancy lost. Lifting the predicate alone buys
  nothing.
- hipfire caps `n_splits` at 8; SGLang/AITER uses `max_split_per_batch` of
  32-64.

## Engine-choice evidence

- **vLLM**: `gfx1201` is in `HIP_SUPPORTED_ARCHS` (`CMakeLists.txt:52`), but
  `AITER_ROCM_ARCH=gfx942;gfx950` (`docker/Dockerfile.rocm_base:37`) — the tuned
  AMD kernels are CDNA-only. RDNA4 gets Triton/PyTorch fallbacks.
- **SGLang**: `AMDGPU_TARGET="gfx942;gfx950"`; the sgl-kernel gate is
  `["gfx942","gfx950","gfx1250"]`, with `docker/patches/sgl-kernel-gfx1151.sh`
  existing purely to lift it for one RDNA3.5 part. gfx1201 is not on it.
- **llama.cpp**: RDNA4 is first-class (`GGML_CUDA_CC_RDNA4`, included in the
  MMA-capability predicates at `common.cuh:328,357`), and `LLM_ARCH_QWEN35`
  loaded and ran this checkpoint family without modification.

## The MTP head — the biggest single lever, and hipfire's file cannot reach it

Qwen3.8 ships a native MTP (NextN) head. It is present in the Unsloth GGUF —
`blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`, layer 64 past the
64 trunk layers, matching `mtp_num_hidden_layers: 1` in the checkpoint header.

**`qwen3.8-27b.mq4` carries the config keys but none of the tensors** — they
were dropped at quantization. hipfire supports MTP (`mtp_head.rs`, 2,616 lines;
`speculation.mtp` defaults to `"auto"`; loads a `.mtp` file built by
`mtp_extract.rs`), but with no `.mtp` file it silently falls through to DFlash.

Independent measurement on this same card (akougkas.io, 2026, ROCm 7.2.4,
131k ctx, 4 slots) puts MTP well ahead of DFlash2:

| drafter | decode | acceptance |
|---|---:|---:|
| none | 27.1 t/s | — |
| **MTP draft 2** | **46.2 t/s** | **66%** |
| DFlash2 (Q8_0, n=5) | 36.8 t/s | 37% |

Their production preset reaches **47.5 t/s single-stream at 131,072 context**
with 67% acceptance and 892 t/s prefill in 27.5 GiB — against hipfire's 7.8 t/s
and 101 t/s here. They also independently confirm two of our findings: KV type
is "purely a trade of memory for quality" (q4_0/q8_0/f16 within noise on speed),
and IQ4_NL reaches **487 GB/s, 76% of the card's 640 GB/s** — against hipfire's
~78 GB/s at long context.

**Caveat on context reach:** they cite llama.cpp issue 27756 — this hybrid emits
an instant EOS beyond ~130k positions on every backend, capping usable context
there. hipfire handled 165,282 tokens without trouble, just slowly.

## What hipfire still has that llama.cpp does not

- **DFlash2** — EAGLE-style hidden-state drafting. Worth ~2x here (tau 2.03 at
  124K, ~3.3 at short context).
- **fwht3 KV** — 31.6% less KV than q8 for +1.64% PPL (measured same day).
- MQ4V2 + the Hessian/GPTQ quantizer and its per-layer kmap.
- The VL sidecar path and its batched prefill (82 s -> 2.1 s).

**The combination is the prize.** llama.cpp's attention with hipfire's DFlash2
projects to ~31 tok/s at 124K against the 7.8 measured today.

## Open

- Whether to port DFlash2 + fwht3 onto llama.cpp, or port llama.cpp's
  long-context attention into hipfire. The former looks cheaper; not decided.
- 250K never completed on hipfire (>30 min timeout; extrapolates to ~83 min
  cold). Unmeasured on llama.cpp.
- Image-turn path unmeasured on both.
