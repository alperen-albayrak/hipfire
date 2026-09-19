# Bonsai 2 27B, the KV tier, and two-lane concurrency on R9700

**Goal:** Serve two independent 262K-token conversations concurrently on a
single 32 GB R9700 (gfx1201), keeping the fwht3 K tier. Today neither half is
possible: the model is too big for two full-context lanes, and the multi-slot
path is hard-wired to Q8 KV.

**Architecture:** Three independent levers that compound, in dependency order —
(1) establish what the fwht3 K tier actually costs against Q8, because
everything below is justified by that number; (2) port PrismML's Bonsai 2 27B,
which is Qwen3.8-27B in a Hadamard-rotated ternary basis, reusing the existing
`TQ2G128` kernels through a rotated sibling dtype; (3) teach the multi-slot KV
path the asym3 tier, which already has its attention kernel and needs its write
and prefill kernels.

**Tech Stack:** Rust (`hipfire-quantize`, `hipfire-dispatch`,
`hipfire-arch-qwen35`, `rdna-compute`, `hipfire-daemon`). Two new HIP kernels
(Task 5, Task 6); everything else reuses kernels already in tree.

**Hardware envelope:** gfx1201 (Radeon AI PRO R9700, 32 GB), single card. Every
number in this plan is for that envelope and does not generalise.

**Related:** [`2026-09-18-vl-multiturn-prefill-plan.md`](2026-09-18-vl-multiturn-prefill-plan.md)
— shares the "make the second turn cheap" goal from the VL side. Independent
work; they compound on the same VRAM budget.

---

## Baseline

### KV cost per token

**The allocation formula, verified against the allocator** — not the tier
helper. `KvCache::new_gpu_asym3_*` (`crates/hipfire-runtime/src/llama.rs:6752`):

```
k_bph  = 4 + head_dim*3/8            = 100 B/head  →  400 B/pos   (asym3 K)
v_bpp  = n_kv_heads*(head_dim/32)*34               → 1088 B/pos   (Q8 V)

KV bytes = max_seq × n_full_attention_layers × (k_bytes_per_pos + v_bytes_per_pos)
```

Two properties of that allocator matter for sizing and were confirmed by code
read, not assumed:

- It allocates **upfront for `max_seq`** (`physical_cap`), not grown per token.
  A conversation's KV footprint is fixed at load, by configuration.
- It allocates **only for full-attention layers**. The `*_filtered`
  constructors take an `is_kv_layer` mask, built at
  `crates/hipfire-arch-qwen35/src/speculative.rs:896` as
  `*t == LayerType::FullAttention`, and substitute a 1-element placeholder for
  the rest. DeltaNet layers carry fixed-size recurrent state that does not grow
  with context — that is what makes a 262K lane askable at all.

At Qwen3.8-27B geometry (H24 / KV4 / head_dim 256):

| K tier | B/pos/layer | with Q8 V | per token (16 layers) | 262,144 tokens |
|---|---:|---:|---:|---:|
| Q8 | 1088 | +1088 | 34.0 KiB | 9.1 GB |
| asym4 / fwht4 | 528 | +1088 | 25.3 KiB | 6.7 GB |
| **asym3 / fwht3** | **400** | **+1088** | **23.3 KiB** | **6.2 GB** |
| asym2 / fwht2 | 272 | +1088 | 21.3 KiB | 5.6 GB |

**The 16-layer count is NOT verified.** It comes from PrismML's "~75% linear /
~25% full" against 64 layers. Read it from the checkpoint's `config.json`
before sizing anything on it — every absolute figure in this plan scales
linearly with it.

### What is robust, and what is not

Separate the two, because they carry different confidence:

**The Q8-over-fwht3 ratio is exact and assumption-free:**

```
2176 / 1488 = 1.4624
```

Both the full-attention layer count and `max_seq` cancel. "Q8 costs ~1.46×
fwht3" holds whatever those turn out to be, and it is the number the asym3
slot-kernel work (Tasks 4–6) is justified by.

**The absolute pair (18.3 GB vs 12.5 GB for two 262K lanes) is an estimate**
that scales with the layer count above. If the true count is 12, the pair
becomes 13.7 / 9.4 GB — same ratio, different budget. Treat the absolutes as
the shape of the problem, not as numbers to plan VRAM against, until the
baseline row below is filled in.

V at Q8 (1088 B) dominates once K drops below it, which is why fwht2 buys so
little over fwht3 and why the V-quant ladder (lloyd4/3/2) exists separately.

### Measured baseline on cachy-01 — TO BE FILLED

Everything above is arithmetic. This row anchors it to the running system, and
should be recorded before any capacity decision:

| field | value |
|---|---|
| model + quant | qwen3.8-27b MQ4V2 |
| model bytes on disk | — |
| `max_seq` configured | — |
| full-attention layer count (from `config.json`) | — |
| KV bytes by the formula above | — |
| VRAM high-water mark, 1 conversation | **~27 GB of 32** (observed, unquantified) |

**The observation is already load-bearing, independent of the layer count.** At
~27 GB with a ~14 GB model, KV plus everything else is ~13 GB **for one lane**.
A second lane at the same context does not fit — it would want ~40 GB.
Replacing a 14 GB model with Bonsai 2's 5.9 GB frees ~8 GB, taking headroom
from ~5 GB to ~13 GB and roughly **doubling the KV budget at the same context
setting**.

That is the two-lane argument made from a measurement rather than from the
estimate tables, and it does not depend on the unverified layer count. It is
the strongest form of the case this plan rests on.

### Quality ladder

`benchmarks/quality-baselines/results/2026-05-31-kv-vquant/24chunk-full-matrix-results.txt`
— qwen3.6-27b.mq4, gfx1100, 24-chunk, all rows with q8-V:

| K tier | KLD | PPL |
|---|---:|---:|
| fwht4 | 0.010640 | 3.4368 |
| **fwht3** | **0.011148** | **3.4366** |
| fwht2 | 0.015053 | 3.4508 |

fwht3 matches fwht4 within noise at 76% the size; fwht2 is a real drop. fwht3
is the knee, which is why it is the production choice.

**That campaign has no q8-K row.** The ladder is calibrated against itself, not
against the tier above it. Task 0 exists to close this.

### Model footprint

| model | bpw | size | source |
|---|---:|---:|---|
| qwen3.8-27b MQ4V2 | — | ~14 GB | estimate, not measured |
| Bonsai 2 PQ2_0 | 2.16 | 7.25 GB | PrismML docs |
| Bonsai 2 PTQ1_0 | 1.76 shipped / 1.72 true | 5.93 GB | PrismML docs |

### Two 262K lanes against 32 GB

| model | KV ×2 @ Q8 | KV ×2 @ fwht3 |
|---|---:|---:|
| MQ4V2 (~14 GB) | 32.3 GB ✗ | 26.5 GB — no headroom |
| Bonsai 2 PQ2_0 (7.25) | 25.6 GB — tight | **19.7 GB** |
| Bonsai 2 PTQ1_0 (5.93) | 24.2 GB | **18.3 GB** |

Excludes DeltaNet state, activation and prefill scratch, the DFlash drafter,
and the vision sidecar. Real headroom is thinner than these totals. The point
of the table is the ordering, not the absolute numbers: **the small model is
what makes two lanes possible, and the fwht3 tier is what makes them
comfortable.**

---

## Grounding facts (verified by code read — reference while implementing)

**Bonsai 1 already ships.** `dtype_from_quant_type`
(`crates/hipfire-arch-qwen35/src/qwen35/weights.rs:236`) maps qt=40 →
`TQ2G128` (ternary, 34 B per 128 = 2.125 bpw) and qt=41 → `BQ1G128` (binary,
18 B per 128). Ingest is **byte-verbatim passthrough**
(`crates/hipfire-quantize/src/pipeline_gguf.rs:498`): "layout is byte-identical
between the GGUF Q2_0 block and hipfire TQ2G128". Quality acceptance lives in
`benchmarks/quality-baselines/results/2026-08-17-spe-ptq/`, scored against
PrismML's own llama.cpp.

**Bonsai 2 is a different container.** It ships `PQ2_0` (2.16 bpw) and
`PTQ1_0` (1.76 bpw), not `Q2_0`/`Q1_0`, and both are in a **rotated basis** —
PrismML's docs describe PTQ1_0 as "Ternary g128 with FP16 group-wise scaling,
blockwise Hadamard rotation", and state both packings need their llama.cpp fork
because the runtime Walsh–Hadamard is not upstream.

**hipfire's existing ternary is UNROTATED.** `dtype_rotation_plan`
(`crates/hipfire-dispatch/src/types.rs:118`) leaves `TQ2G128` on the `_` arm →
`RotationPlan::None`, and `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:1587`
says so: "Unrotated, so they take the plain-rmsnorm activation path". Feeding
rotated-basis weights through this is **silent garbage, not an error** — the
same failure mode `types.rs:130` already warns about.

**But the rotation machinery exists.** `RotationPlan::FwhtG128` is production
for `MQ4G128`; the whole MQ family is FWHT-256-rotated. hipfire already has
what forces PrismML to ship a fork.

**And the sibling-dtype pattern is established.** `MQ2G256Lloyd` /
`MQ2G256LloydU` are rotated/unrotated siblings with "byte-identical 72 B/group
layout, so every kernel binds" (`crates/hipfire-arch-maple/map.md`). The
`types.rs:130` comment explicitly demands that whoever adds the next variant
decide which side of the rotation line it is on, rather than letting it fall
through to `None`.

**Ternary GEMM already routes to WMMA on gfx1201.** `gemm_lowbit_prefill`
(`crates/rdna-compute/src/gemv.rs:5733`) forwards to
`gemm_tq2g128_wmma` whenever `arch_caps.has_wmma()`, measured "6.6× vs scalar
at M=17408 N=128". gfx1201 qualifies. No new GEMM work for either packing.

**Continuous batching is built, not hypothetical.** `ContinuousBatchScheduler`
is constructed on load at `crates/hipfire-daemon/src/main.rs:2076` from
`staging.slots` / `staging.lane_capacity`; `forward_slots.rs` does batched
attention and a batched lm_head across `n_slots`; the load reply advertises
`continuous_batch_slots` and `continuous_batch_lane_capacity`; there is a
`bench_concurrency` CLI and a test suite at
`crates/hipfire-engine/tests/continuous_batch.rs`.

**Do not confuse it with the slot backend.** `crates/hipfire-daemon/src/slots.rs`
is an "Experimental multi-slot daemon backend — **alternate model owner, NOT a
continuous-batching mode**". It advertises `continuous_batch_capable: false`,
rejects vision models, and its workers "serialize on engine submit". It gives
no parallelism and is not the target of this plan.

**Why the multi-slot path is Q8-only — addressing, not numerics.**
`crates/hipfire-arch-qwen35/src/forward_slots.rs:32`:

> `SlotPool`'s per-slot addressing is documented as a Q8_0 ABI (asym3 is
> explicitly exempted because its K/V strides differ and it cannot share
> `k_base`/`v_base`)

and at line 36: "The KV cache tier is unconditionally Q8_0 for EVERY layer this
file drives, dense or MoE — slot addressing is a KV-cache-tier property, not a
weight-quant property."

**The descriptor ABI already supports two strides.** `KvSlotDesc`
(`crates/rdna-compute/src/kv_slots.rs:26`) carries **separate** `k_base` and
`v_base`. The exemption is about the arena *builders* assuming a uniform
34-byte block layout — which is exactly why `build_asym3_k_arena`
(`kv_slots.rs:158`) had to be written as a separate generator, "found necessary
empirically while running the Task 7 harness".

**Slot-kernel inventory.** Q8 has five, asym3 has one:

| | Q8 | asym3 |
|---|---|---|
| KV write | `kv_cache_write_q8_0_batched_slots` (attention.rs:1558) | **missing** |
| flash tile attn | `attention_flash_q8_0_batched_masked_slots` (5029) | ✅ `attention_flash_asym3_batched_masked_slots` (7599) |
| flash prefill | `attention_q8_0_flash_prefill_slots` (3031) | **missing** |
| flash prefill WMMA | `attention_q8_0_flash_prefill_wmma_slots` (3308) | **missing** |
| LDS decode/verify | `attention_q8_0_kv_batched_masked_slots` (2407) | missing — low priority |

The asym3 tile kernel takes `tree_bias: Option<..>`, so unlike the FA2 fast
path it does **not** exclude tree-verify. The LDS kernel is "context capped
well under the ~16k LDS ceiling", so at 262K the tile path is what runs anyway.

**Non-slot asym3 siblings exist for both missing kernels:**
`kv_cache_write_asym3_batched` (attention.rs:7382), `kv_cache_write_asym3_fused`
(5703). The port is adding slot-descriptor addressing — the same delta that
produced `attention_flash_asym3_batched_masked` → `..._slots`.

**The test harness for slot isolation already exists.**
`crates/rdna-compute/examples/test_batched_attn_slots.rs` carries a negative
control (corrupt every descriptor to slot 0), a NaN-poison isolation check on
neighbouring slots, and `test_poison_is_live` proving the poison mechanism is
not inert. New slot kernels land into this harness.

**The eval harness already sweeps K tiers.**
`crates/hipfire-runtime/examples/eval_hipfire.rs:99` accepts
`--kv-mode q8|asym2|asym3|asym4|fwht2|fwht3|fwht4|f32|f16` and
`--kv-v q8|lloyd2|lloyd3|lloyd4`. Task 0 needs no new code.

**gfx1201 fast paths are already default ON** — `HIPFIRE_GFX12_FA2_PREFILL`
(GQA-fused FA2 prefill, with a dedicated fwht3-K variant at exactly
H24/KV4/D256) and the `HIPFIRE_GFX12_MQ4V2_FP8_*` family, per
`docs/env-vars.md:154`. There is no disabled-fast-path win to collect here.
Note both FA2 arms require `io.tree_bias.is_none()`
(`crates/hipfire-dispatch/src/families/attention.rs:1631,1668`), and the fwht3
arm additionally fails closed under graph capture because its in-place Q
rotation is not replay-idempotent.

---

## Task 0: measure fwht3 against Q8 K (BLOCKING, no new code)

**Goal:** The number this entire plan is justified by. Every task below trades
quality for capacity on the assumption that fwht3's cost is small. Nobody has
measured it.

**Files:** none — `eval_hipfire` already does this.

**Do:** Run the same model and reference at `--kv-mode q8 --kv-v q8` and
`--kv-mode asym3 --kv-v q8`, on gfx1201, `--scoring-mode prefill`. Add the q8-K
row to the 2026-05-31 matrix so the ladder is calibrated against its top rung.
Use qwen3.8-27b so the number applies to the model actually being served — the
existing matrix is qwen3.6-27b on gfx1100.

**Done when:** a q8-K/q8-V KLD and PPL exist next to the fwht3 row, measured on
gfx1201, committed under `benchmarks/quality-baselines/results/`.

**Decision it gates:** if fwht3's cost over Q8 is comparable to its cost over
fwht4 (~0.0005 nats), Tasks 4–6 are clearly worth building. If it is large,
the honest answer is to serve two lanes at Q8 with a smaller model and skip
them — Bonsai 2 alone already fits that (24.2 GB).

**Risk if skipped:** high, and of the worst kind — three tasks of kernel work
justified by an unmeasured assumption.

---

## Task 1: PQ2_0 ingest as a rotated ternary sibling

**Goal:** Load Bonsai 2 at 7.25 GB with zero new kernels.

**Files:** `crates/hipfire-quantize/src/pipeline_gguf.rs` (tensor-type match),
`crates/hipfire-dispatch/src/types.rs` (`dtype_rotation_plan`,
`dtype_post_rotation_variant`), `crates/hipfire-arch-qwen35/src/qwen35/weights.rs`
(`dtype_from_quant_type`), `crates/hipfire-dispatch/src/families/gemm.rs`.

**Do:** Add `TQ2G128R` — the rotated sibling of `TQ2G128`, exactly as
`MQ2G256Lloyd`/`MQ2G256LloydU` are siblings, but with the rotation flag the
other way:

- `dtype_rotation_plan(TQ2G128R) => RotationPlan::FwhtG128` (stated
  explicitly in the match, never via the `_` arm — `types.rs:130` demands it)
- `dtype_post_rotation_variant(TQ2G128R) => GemvVariant::Prerotated`
- every kernel key that accepts `TQ2G128` accepts `TQ2G128R`, byte-for-byte —
  the packing is identical, only the basis differs
- new qt number for the HFQ carrier; passthrough ingest keyed on the GGUF
  `PQ2_0` tensor type

**First, verify the block layout.** The 34 B/128 assumption comes from bit-width
arithmetic (2.125 bpw against PrismML's stated 2.16), **not** from reading a
tensor. Dump one PQ2_0 tensor and confirm bytes-per-block, group size, and
scale dtype before writing any of the above. If it differs, this task grows a
repack step and Task 3's estimate is also suspect.

**Also confirm the rotation contract:** FWHT size (the `FwhtG128` plan rotates
in 128-blocks; PrismML say "blockwise Hadamard" without stating the block),
sign convention, and whether normalisation is folded into the group scale. A
size or sign mismatch produces fluent wrong text, not an error.

**Done when:** `hipfire-quantize` produces a `.hfq` from the PQ2_0 GGUF, it
loads, and it generates coherent text on the 5-genre serve battery.

---

## Task 2: is Bonsai 2 actually good? (GATE)

**Goal:** Decide whether anything below is worth building.

**Files:** `benchmarks/quality-baselines/results/2026-09-XX-bonsai2/`.

**Do:** Two measurements, both with precedent in tree:

1. **Port fidelity** — `eval_hipfire` against a reference built from PrismML's
   own llama.cpp fork running the same GGUF, the method
   `results/2026-08-17-spe-ptq/README.md` established for Bonsai 1. This
   measures *our port*, not the model.
2. **Model quality** — Bonsai 2 vs qwen3.8-27b MQ4V2, same reference, same
   KV tier. This measures what the 9× compression costs.

Then run real agent traffic through it. PrismML claim 98.2% retention; the
2026-08-17 campaign **withdrew** its earlier Bonsai KLD numbers over a KV
confound, so treat vendor retention figures as a hypothesis.

**Done when:** both numbers are committed with the KV tier and scoring mode
recorded in the fixture, and there is a written go/no-go.

**Risk if skipped:** building Task 3's unpack kernel and three slot kernels for
a model that is not good enough to serve.

---

## Task 3: PTQ1_0 dense trit unpack

**Goal:** 5.93 GB instead of 7.25 GB — the 1.35 GB that turns "tight" into
"comfortable" in the two-lane table.

**Files:** `crates/rdna-compute/src/kernels.rs` (new unpack),
`crates/rdna-compute/src/gemv.rs`, `crates/hipfire-quantize/src/pipeline_gguf.rs`.

**Do:** PTQ1_0's "1.72 bpw true ternary" is consistent with **5 trits per byte**
in base-3 (3⁵ = 243 ≤ 256 → 1.6 bpw) plus an FP16 group scale per 128
(+0.125) = 1.725. Confirm against a real tensor before implementing. The unpack
is a base-3 decode: `t[i] = (byte / 3^i) % 3 - 1`, either by a 256×5 LUT in LDS
or by the multiply-shift sequence.

Everything downstream is Task 1's `TQ2G128R` — same rotated basis, same group
scales, same GEMM. Only the decode differs.

**Done when:** PTQ1_0 loads and its logits match the PQ2_0 build of the same
checkpoint within the batched-vs-per-token float tolerance (~7.5e-3, per
`test_spec_rope_phase_bias_parity`'s finding). The two packings encode the same
weights; a larger divergence means the unpack is wrong.

---

## Task 4: `forward_slots.rs` KV tier is a parameter, not a constant

**Goal:** The plumbing half of asym3-in-slots, landed before the kernels so
they have a caller.

**Files:** `crates/hipfire-arch-qwen35/src/forward_slots.rs`,
`crates/rdna-compute/src/kv_slots.rs` (`SlotPool` arena sizing).

**Do:** Replace the unconditional Q8_0 tier with a `KTier` carried on the slot
pool. `KTier::k_bytes_per_pos` already computes asym3's stride correctly — the
number exists and is not threaded through. `KvSlotDesc` needs no change:
`k_base` and `v_base` are already independent.

Keep Q8 the default and the only *enabled* tier until Tasks 5 and 6 land, so
this is a refactor with no behaviour change.

**Done when:** the existing Q8 multi-slot tests pass byte-identically, and the
tier is reachable as a parameter.

---

## Task 5: `kv_cache_write_asym3_batched_slots`

**Goal:** Without this there is no multi-slot fwht3 at all — every decode step
writes KV.

**Files:** `crates/rdna-compute/src/attention.rs`,
`crates/rdna-compute/src/kernels.rs`.

**Do:** Port `kv_cache_write_asym3_batched` (attention.rs:7382) to
slot-descriptor addressing, the same delta that produced
`attention_flash_asym3_batched_masked` → `..._slots`. Per-(position, kv_head)
record is `[4-byte cnorm f32][packed 3-bit body]`, `k_bytes_per_head = 4 +
head_dim*3/8` = 100 at hd=256.

**Watch the alignment hazard `build_asym3_k_arena` documents:** 100 is not a
multiple of 34, so a `cnorm` read lands at an offset the Q8 arena generator
never constrains. That is how a NaN appeared to leak between slots when it had
actually been non-finite all along — and `assert_close` was silently vacuous
because `NaN > worst` is false. Use `build_asym3_k_arena`, not `build_arena`.

**Done when:** it passes `test_batched_attn_slots` including the negative
control and the poison-isolation layer, across the adversarial shape set
(ragged tiles, mixed M, unequal per-slot context, slot counts 1..8).

---

## Task 6: `attention_asym3_flash_prefill_slots`

**Goal:** Batched prefill for an asym3 lane. Without it, prefill either falls
back per-token or forces Q8.

**Files:** `crates/rdna-compute/src/attention.rs`, `kernels.rs`.

**Do:** Port `attention_q8_0_flash_prefill_slots` (attention.rs:3031) to the
asym3 K layout. Its own hazard, per the harness: "a single tile can span
several query rows of one slot, so ragged-tile-at-slot-boundary is its own
hazard".

Decide explicitly whether to port the WMMA sibling
(`attention_q8_0_flash_prefill_wmma_slots`) — prefill throughput on gfx1201 is
the reason the FA2 kernels exist, and a non-WMMA asym3 prefill may erase the
capacity win with a latency loss. Measure before committing to one.

**Done when:** it passes the same harness layers, and prefill throughput for a
single asym3 slot is within noise of the current non-slot asym3 prefill.

---

## Task 7: two 262K lanes

**Goal:** The actual objective.

**Files:** daemon load config (`slots`, `lane_capacity`).

**Do:** `slots: 2, lane_capacity: 262144` against Bonsai 2 with the asym3 tier.
Confirm `continuous_batch_capable: true` is advertised, then measure where it
actually runs out — the estimate table excludes DeltaNet state, scratch, and
the drafter, and one of those is likely to bind first.

**Done when:** two independent 262K conversations are resident, neither pays
re-prefill on its second turn, and the measured VRAM high-water mark is
recorded.

**Note on what this buys:** two resident lanes mainly means **neither pays
re-prefill**, not 2× throughput. For an agent workload that is the win that
matters — and it is the same win as the VL prefix-reuse work, from the other
direction.

---

## Sequencing

```
Task 0 (measure fwht3 vs Q8)  ← BLOCKING: justifies Tasks 4-6
   │
Task 1 (PQ2_0 ingest) ──► Task 2 (is it good?)  ← GATE: justifies everything
                               │
                               ├──► Task 3 (PTQ1_0, 1.35 GB)
                               │
                               └──► Task 4 (tier plumbing)
                                       └──► Task 5 (KV write)   ← both required
                                       └──► Task 6 (prefill)    ← both required
                                               └──► Task 7 (two lanes)
```

Tasks 0 and 1 are independent and can run in parallel. Nothing after Task 2
starts before its go/no-go.

**PR boundaries:** Task 1+3 (the Bonsai port) and Tasks 4-6 (asym3 slots) are
independent and separately reviewable. Task 0 and Task 2 are benchmark commits.

---

## Validation

- **Task 0 and 2 are measurements**, not code — their output is a committed
  fixture with the KV tier and scoring mode recorded, per the 2026-08-17
  campaign's own correction about confounded KV.
- **Rotation correctness (Task 1) is the silent failure class.** A wrong FWHT
  size or sign produces fluent wrong text. The 5-genre serve battery catches
  gross breakage; the llama.cpp cross-reference catches subtle breakage. Both.
- **Slot isolation (Tasks 5, 6) is the other silent class.** Cross-slot leakage
  looks like a quality regression, not a crash. The existing harness has the
  negative control and the liveness check — use them; do not add a new test
  path that lacks them.
- Repo gates as usual: `scripts/fmt-changed.sh <base>`,
  `scripts/leanup-ratchets.sh` green, `RATCHET-RAISE:` trailer if a ceiling
  moves.

---

## Out of scope

- **Vision.** Bonsai 2 ships its own mmproj (0.47B tower, HQQ-4bit 0.63 GB or
  BF16 0.93 GB). Whether it drops into the existing VL sidecar path is a
  separate question, and the VL plan is mid-flight.
- **DFlash on Bonsai 2.** No draft model exists for it. Ternary decode
  throughput is also unmeasured — it may not need one.
- **`BQ1G128` / binary Bonsai 2.** No binary variant announced.
- **More than 2 lanes.** The scheduler's fixture uses 4; this plan targets 2 at
  full native context, which is the harder constraint.
- **`experimental_multi_slot`.** Explicitly not continuous batching; not a path
  to this goal.

---

## Open questions

1. **PQ2_0's actual block layout.** 34 B/128 is arithmetic, not evidence.
   Blocks Task 1.
2. **The Hadamard contract** — block size, sign convention, normalisation.
   Blocks Task 1, and gets no error if wrong.
3. **Full-attention layer count** for Qwen3.8-27B. Every *absolute* capacity
   number scales with it. The Q8/fwht3 ratio (1.4624) does not — see
   "What is robust, and what is not".
4. **The cachy-01 baseline row** — `max_seq`, model bytes, and the measured
   VRAM high-water mark. Cheapest item here and it anchors everything else;
   the ~27 GB observation is currently unquantified.
5. **Whether asym3 prefill needs the WMMA sibling** to avoid trading a
   capacity win for a latency loss (Task 6).
6. **What actually binds first** in the two-lane budget — the estimate excludes
   DeltaNet state, prefill scratch, and the drafter.
7. **Long-context decode on gfx1201 is unmeasured.** The only long-context
   decode data in tree is gfx1100 / qwen3.6 / fwht3 / no DFlash: 35.2 tok/s at
   32,649 ctx
   (`benchmarks/quality-baselines/results/2026-05-31-kv-vquant/longctx-decode-ab.txt`).
   This plan sizes 262K lanes without knowing whether decode stays usable at
   length. For scale, Inco's SPLASH reports 54 tok/s at 32K on Qwen3.8-27B on
   M3+ Macs, against hipfire's ~140 tok/s at short context on gfx1201 — so the
   short-context lead is large and the long-context behaviour is simply
   unknown. Measure before committing to the context ceiling.
