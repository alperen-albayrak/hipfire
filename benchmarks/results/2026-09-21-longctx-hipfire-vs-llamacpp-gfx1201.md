# Long-context: hipfire vs llama.cpp on one R9700 (gfx1201), 2026-09-21

**One-line result: at 124K context llama.cpp is 5.6x faster at prefill and 4.0x
faster at decode than hipfire, on the same GPU, in the same hour — while
carrying a 12% larger model and reading 46% more KV.**

hipfire is *faster* at short context. It collapses at long context; llama.cpp
degrades gently. That difference is the finding.

## Numbers

Both engines, cachy-01, AMD Radeon AI PRO R9700 (gfx1201, 32 GB), ROCm 7.2.4.
llama.cpp pinned to device 0 (`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`)
so the 760M iGPU could take no layers — re-running pinned changed nothing
(563.51 vs 563.49, 15.49 vs 15.50), confirming it never contributed.

| | llama.cpp | hipfire | ratio |
|---|---:|---:|---|
| prefill @124K | **563.5 tok/s** | 101.2 tok/s | **5.6x** |
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

llama.cpp wins by 4-5.6x despite all of it.

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
