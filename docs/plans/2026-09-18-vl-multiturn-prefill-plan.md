# VL Multi-Turn Prefill & Position Continuity — Implementation Plan

**Goal:** Make an image turn cost what a text turn costs. Today a VL request
prefills per-token at ~33 tok/s, discards the conversation KV on every turn, and
decodes without DFlash. Fix the three together: batch the VL prefill, carry the
M-RoPE position cursor across turns, and let the existing prefix cache apply to
conversations that contain images.

**Architecture:** Promote the M-RoPE position cursor to *primary* conversation
state (`mrope_cursor`, beside `seq_pos`) instead of deriving it per request;
extend `forward_prefill_batch` with an embedding override and an explicit
per-token position array; teach the `conversation_tokens` LCP match to compare
image spans by content id. This is the llama.cpp `mtmd` / `server_tokens`
design, which solves the same problem — see § Prior art.

**Tech Stack:** Rust (`hipfire-generate`, `hipfire-arch-qwen35`,
`hipfire-arch-qwen35-vl`, `hipfire-daemon`). No new HIP kernels: the batched
prefill kernel already exists and is used by the text path.

**Predecessor:** `docs/plans/completions_vision.md` (Phase 1 — image input over
`/v1/chat/completions`). Its "single-turn only" limitation was never enforced by
the promised HTTP 400 guard; multi-turn framing landed instead in
`fix(vision): frame the whole conversation on VL image turns`, which is what
makes prefill cost the dominant issue this plan addresses.

---

## Measured baseline

gfx1201 (Radeon AI PRO R9700, 32 GB), `qwen3.8:27b` MQ4V2 + `fwht3` KV,
DFlash draft `qwen38-27b-dflash-mq4`, 2026-09-18, on the multi-turn framing fix.

| workload | prompt | TTFT | prefill | decode |
|---|---|---|---|---|
| text, ~200-tok history | 373 | 0.57 s | 655 tok/s | 139.6 tok/s (dflash on) |
| image, ~200-tok history | 406 | 12.3 s | 33.0 tok/s | 32.1 tok/s (dflash off) |
| text, ~1400-tok history | 2654 | 4.2 s | 628 tok/s | — |
| image, ~1400-tok history | 2687 | **82.0 s** | 32.8 tok/s | — |

Prefix reuse, identical request issued twice:

| | prompt | cached | TTFT |
|---|---|---|---|
| text, cold | 2954 | 0 | 4.73 s |
| text, repeat | 2954 | **2304** | **1.15 s** |
| text, conversation extended by one turn | 2991 | **2304** | 1.19 s |
| image, cold | 1487 | 0 | 45.3 s |
| image, repeat (byte-identical) | 1487 | **0** | 45.3 s |

Targets: image prefill ≥ 500 tok/s; image repeat/extension reuses its prefix
(cached > 0); image decode ≥ 100 tok/s once Task 6 lands.

---

## Grounding facts (verified by code read — reference while implementing)

**Per-token VL prefill.** `crates/hipfire-generate/src/vision.rs` (~line 1311):
"VL prefill is per-token (`forward_scratch_embed` isn't batched), so we advance
`m.seq_pos` in-loop". The loop calls `qwen35::forward_scratch_embed_mrope` for
image-pad positions and the token variant otherwise. This is the entire 19×
prefill gap — the text path calls `qwen35::forward_prefill_batch`.

**`forward_prefill_batch` signature** (`crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:845`):
`(gpu, weights, config, tokens, start_pos, kv_cache, dn_state, scratch,
hidden_rb, per_token_hidden_out, gdn_tape, tree_verify)`. It has **no**
embedding-override and **no** position-array parameter — those are the two
additions in Task 2. Note it already carries `hidden_rb`, which Task 6 needs.

**The cursor is computed and then discarded.**
`crates/hipfire-arch-qwen35-vl/src/mrope.rs:57` `build_mrope_positions` advances
`cursor += 1` per text token but
`cursor += max(grid_h, grid_w) / spatial_merge_size` per image span, then
returns `rope_delta = max_pos + 1 - n_tokens`, documented as "Added to the
running sequence length to get decode-step positions". Nothing persists it
across requests.

**The bail.** `vision.rs` `build_vl_mrope_ctx` refuses `base > 0` with
"cross-turn mrope cursor continuity not modelled", and its comment states the
correct semantics: "HF would resume a later turn at `previous_max + 1` (i.e.
`base` + the earlier turn's rope_delta)". Everything after the bail already
accepts an arbitrary span offset — the splice validator finds the pad run by
search and only requires it to be one contiguous run, which the multi-turn
framing fix already exercises (observed `span start=713`, `base=0`).

**The force-reset.** `crates/hipfire-daemon/src/main.rs`, VL dispatch arm: when
`seq_pos > 0` it logs "non-zero seq_pos (N) at VL dispatch — resetting
conversation" and clears `seq_pos`, `conversation_tokens`, both checkpoint rings
and recurrent state. Its rationale is that the live KV was written with 1-D
positions and splicing visual tokens into it "would produce garbage". That
rationale dissolves once the whole conversation is positioned under one mrope
cursor (Task 1) — a text run under mrope is `[cursor; 3]` on all three axes,
which is numerically identical to 1-D RoPE.

**The prefix cache exists and works — for text.** The daemon keeps
`m.conversation_tokens` and matches on longest common prefix
(`prompt_frame::continuation_suffix` documents appending as "the precondition
for prefix reuse"). Measured above: 2304 tokens reused. VL never benefits
because the force-reset clears `conversation_tokens` first.

**The speculator is already built for VL models.**
`crates/hipfire-loader/src/lib.rs` (~2201) selects `DflashSpeculator` whenever a
DFlash draft is loaded, arch-generically, and parks it on
`LoadedModel.speculator`. `generate_vl` only ever *resets* it. The obstacle is
seeding: `seed_target_hidden_from_prompt`
(`crates/hipfire-arch-qwen35/src/speculative.rs:8365`) re-prefills from **token
ids** via `forward_prefill_batch`, which cannot reproduce a VL prompt because
image positions are embeddings with no token spelling.

### Prior art — llama.cpp `mtmd`

Verified against `ggml-org/llama.cpp` @ 2026-09-18. It solves this by keeping
token index and position as two coordinate systems and converting explicitly:

- `tools/server/server-common.cpp` `pos_from_tokens()` walks the prompt: text →
  `pos++, idx++`; media → `pos += n_pos, idx += n_tok`. `size_up_to_pos()` is
  the inverse.
- `tools/mtmd/mtmd.cpp` `mtmd_image_tokens_get_n_pos()` returns
  `max(nx, ny)` for M-RoPE — **the same advance hipfire computes.**
- `tools/mtmd/mtmd-helper.cpp` carries it: `n_past += mtmd_input_chunk_get_n_pos(chunk)`,
  across chunks and across turns. No per-turn reset, and no equivalent of our
  `base > 0` bail — `n_past` is correct by construction.
- Prefix matching (`server_tokens::get_common_prefix`) stores media as
  `LLAMA_TOKEN_NULL` placeholders plus a `map_idx_to_media` side table, and at a
  media slot compares chunk **id** and token count:
  `if (id_ai == id_bi && n_tok_a == n_tok_b) { i += n_tok_a - 1; continue; }`.
- Batching and M-RoPE are one mechanism: `decode_embd_batch` with
  `n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1` and
  `set_position_mrope_2d()` submits a whole image as one batch with an explicit
  per-token position array. There is no per-token image path.

Takeaway: `rope_delta` bookkeeping is a workaround for treating position as a
quantity derived from a token count. Tracking position directly removes it.

---

## Task 0: M-RoPE cross-turn parity oracle (BLOCKING, CPU-only)

**Goal:** A reference oracle for multi-turn, multi-image position sequences
*before* any behaviour changes. Tasks 1 and 5 can silently mis-position every
token after an image — degraded output, no error, no crash. Nothing in Tasks 1/5
merges without this.

**Files:** `crates/hipfire-arch-qwen35-vl/src/mrope.rs` (tests),
`benchmarks/vision/` (fixture).

**Do:** Dump the HF reference implementation's `get_rope_index()` (the
Qwen-VL conditional-generation class matching this artifact) for a fixed
set of conversations — text-only; one image at turn 0; image at turn 2; two
images in different turns; image last — as JSON fixtures of
`(positions[3][n], rope_delta)`. Add a pure-CPU test that walks the same
conversations through `build_mrope_positions` with an accumulating base and
asserts exact equality per axis.

**Done when:** the fixtures exist, and the current code passes the single-turn
cases and *fails* the cross-turn ones (proving the oracle has teeth).

**Risk if skipped:** high — this is the only defect class in the plan that
produces plausible-looking wrong output rather than an error.

---

## Task 1: `mrope_cursor` as conversation state (CPU, unit-tested)

**Goal:** Position becomes primary state, not a per-request derivation.

**Files:** `crates/hipfire-loader/src/lib.rs` (`LoadedModel`),
`crates/hipfire-generate/src/vision.rs` (`build_vl_mrope_ctx`),
`crates/hipfire-generate/src/common.rs` (reset paths).

**SUPERSEDED — done differently, and better (landed 2026-09-18).** The plan
called for storing `mrope_cursor` on `LoadedModel`. Rejected on contact with
the code: there are **17** `seq_pos = 0` sites and **76**
`conversation_tokens.clear()` sites, and one missed reset mis-positions every
token after an image with no error. A stored cursor is a stale-state bug
waiting for a careless edit.

**Derived instead**, as llama.cpp's `pos_from_tokens()` does: the whole
conversation is reframed on every VL request, so
`mrope::cursor_at_token(upto, spans, merge)` is a pure function of the request
with no cross-request state and nothing to reset. Proven against
`build_mrope_positions` at end-of-prompt and at every span boundary across
two-image layouts (`cursor_agrees_with_builder_at_every_boundary`), plus an
exact-resumption test.

**The `base > 0` guard STAYS**, contrary to the original plan. The live caller
passes `m.seq_pos` — a token count — which is sound only because the daemon
force-resets it. Removing the guard before a caller passes a real cursor turns
a loud refusal into silent mis-positioning. It is removed in Task 5, with the
caller, not before.

---

## Task 2: embedding override + position array on `forward_prefill_batch`

**Goal:** One batched prefill entry point that can express "these positions
carry these embeddings, at these 3-axis positions".

**Files:** `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
(`forward_prefill_batch`, `forward_prefill_batch_with_pbs`).

**Every piece of this already exists** (verified 2026-09-18 by code read) — the
task is wiring, not kernel work:

- **The batched M-RoPE kernel is written and has ZERO callers.**
  `Gpu::rope_mrope_halfsplit_f32_batched` (`crates/rdna-compute/src/norm.rs:1387`)
  over `kernels/src/rope_mrope_halfsplit_batched.hip`. It already takes
  `positions` as `[batch_size][3]` row-major, a `pos_offset: i32` base — i.e.
  the cross-turn cursor — and `section: [usize; 3]`, and its axis rule
  (`m==1 && i < 3*sec_h → H`, `m==2 && i < 3*sec_w → W`, else `T`) matches
  `mrope_axis_for_freq`. Somebody built this for exactly this job and stopped.
- **Embedding override into the batch already exists.** `MaskEmbedOverride
  { slot, embed }` (`qwen35/config.rs:89`) is applied in
  `batch_chunk_embed_tokens` (`prefill.rs:~4192`) by `memcpy_htod_offset` into
  `pbs.x_batch` *after* the embedding lookup. Everything downstream — attention,
  DeltaNet, FFN — consumes `x_batch` and is agnostic to provenance, which is
  what makes the recurrent path a non-issue. One image is a contiguous span, so
  generalizing `slot: usize` to a span is one wider memcpy, not N writes.
- **The fused FA+rope prep is not on the critical path here.**
  `fa_prep_fused_ok` requires `fusion == DflashFusionCtx::ChainVerify &&
  gpu.arch_caps.is_gfx1100()`. On gfx1201 the else branch (separate
  deinterleave + standalone rope) always runs, so only ONE rope site needs the
  mrope branch. Still add a guard so mrope and `ChainVerify` fusion cannot
  combine on gfx1100.
- **A new position buffer is needed.** `pbs.rope_positions` is
  `alloc!(&[max_batch], DType::F32)` (`qwen35/batch.rs:291`) — one scalar per
  token. M-RoPE needs three `i32` per token, so add a sibling
  `mrope_positions` sized `max_batch * 3`.

**Do:** Add a `mrope_positions` buffer to `PrefillBatchScratch`; generalize
`MaskEmbedOverride` to a contiguous span (`slot`, `rows: &[f32]` of
`len == n * dim`); add `mrope: Option<MropeBatch<'_>>` (positions + `pos_offset`
+ `section`) to `forward_prefill_batch*`, dispatching to
`rope_mrope_halfsplit_f32_batched` when supplied and the existing
`rope_partial_interleaved_f32_batched` when not. Keep every existing call site
compiling by passing `None` — prove byte-identical text output before touching
VL.

**Done when:** text prefill is byte-identical at the logits level with both new
parameters `None`, and a GPU unit test shows a batched run with an override at
position *k* matches a per-token `forward_scratch_embed_mrope` run for the same
input.

**Risk:** touches the hot text path. Mitigate by landing this task on its own
and running the existing Redline/golden fixtures before Task 3.

---

## Task 3: batched VL prefill in `generate_vl`

**Goal:** Delete the per-token prefill loop. This is the 19×.

**Files:** `crates/hipfire-generate/src/vision.rs`.

**Do:** Replace the `for &token in prompt_tokens.iter()` loop with a single
`forward_prefill_batch` call carrying the visual rows as `embed_override` at the
image-pad span and the mrope positions as `positions`. Keep the abort poll at
chunk granularity (`serve.multi_slot_prefill_chunk`-sized sub-batches) so
mid-prefill cancellation still lands on the canonical cancelled pair — the
current per-token loop polls `check_abort` every token, and chunking is the
replacement, not removal. Keep `maybe_evict` / `maybe_downshift` on chunk
boundaries.

**DONE — measured on gfx1201, 2026-09-18.**

| workload | before | after |
|---|---|---|
| image, ~200-tok history | 12.3 s TTFT, 33.0 tok/s | **0.56 s, 730 tok/s** |
| image, ~1400-tok history | **82.0 s** TTFT, 32.8 tok/s | **2.14 s, 1253 tok/s** |
| image prefill, 64px / 512px | 33 tok/s | 606 / 795 tok/s |

**38× on the 1400-token case.** Correctness held: the multi-turn recall test
passes (reads the image AND recalls the conversation), image descriptions match
the pre-change wording at both 64px and 512px, and prompt token counts are
unchanged (85 vs 84, 277 vs 276, 799 vs 799) so framing is untouched.

Text path unregressed — decode 140.0 tok/s against a 139.6 baseline, DFlash
still engaged, prefill 387–1017 tok/s. That was the regression this change
could most plausibly have caused, via `plain_ar_graph_eligible`.

---

## Task 4: image-aware prefix match (CPU, unit-tested)

**Goal:** An unchanged image earlier in a conversation is a cache hit.

**Files:** `crates/hipfire-generate/src/ar.rs` (the live LCP),
`crates/hipfire-generate/src/vision.rs`, `crates/hipfire-loader/src/lib.rs`
(`LoadedModel` conversation state).

**Corrections from the code read (2026-09-18):**

- **The prefix match is NOT in the daemon.** It lives in `ar.rs:~3174` — a
  plain token walk, `while lcp < max_match && m.conversation_tokens[lcp] ==
  rendered[lcp]` — on the text/AR path. `generate_vl` has no LCP at all; it
  force-resets instead. So this task adds prefix reuse to VL, it does not
  merely extend a shared one.
- **A token-only LCP is not merely insufficient, it is WRONG for images.**
  Image positions in `conversation_tokens` are `image_pad_id` repeated
  `n_visual` times, so two *different* images of the same grid size produce
  byte-identical token runs and a token walk matches them happily. Content-id
  comparison is a correctness requirement, not an optimisation.
- **Staleness must fail safe, not fail silent.** Storing spans alongside
  `conversation_tokens` reintroduces the problem Task 1 avoided: 76 clear
  sites, one missed = stale spans. Do NOT rely on remembering to clear them.
  Record the token length the spans describe and require it to equal
  `conversation_tokens.len()` at lookup; a mismatch means "no reuse" and falls
  back to today's full re-prefill. A missed clear then costs performance, never
  correctness. (llama.cpp avoids this structurally, by making `server_tokens`
  own the token vector and the media map together — the cleaner fix, at the
  cost of touching every `conversation_tokens` user.)

**Do:** Record image spans alongside `conversation_tokens` as
`(start, len, content_id)`. Extend the LCP walk: at a span start, match only if
`content_id` and `len` both agree, then skip the whole span — llama.cpp's
`get_common_prefix` arm, transliterated. A mismatch truncates the prefix there,
as today.

**`content_id` must not be a hash of the image bytes alone.** vLLM's
`MultiModalHasher` folds `model_id` plus `hash_factors` into the key, with the
explicit note that "model output depends on the current modality's hash
factors". The same bytes preprocessed under a different config yield different
visual tokens, so a bytes-only key can false-hit and splice embeddings that do
not match the retained KV — silent wrongness of the same family as a bad mrope
cursor. Hash: source bytes (pre-decode, as vLLM does — cheaper and immune to
decoder-version drift) **plus** the resolved grid dims, `spatial_merge_size`,
the vision sidecar's identity/sha, and `image.decode` mode. Any future
preprocessing knob must be added to this key; note that obligation next to the
config it touches.

**Done when:** unit tests cover identical image, changed image, image moved to a
different turn, and text-only, with no GPU.

---

## Task 5: drop the force-reset; prefill the suffix only

**Goal:** The payoff — image turns stop re-prefilling the conversation.

**Files:** `crates/hipfire-daemon/src/main.rs` (VL dispatch arm).

**Do:** Replace the unconditional `seq_pos > 0` reset with the text path's
logic: compute the common prefix (Task 4), keep that KV, prefill only the
suffix. Retain the reset as the *fallback* for a prefix miss, and keep it
unconditional under a kill switch (`HIPFIRE_VL_PREFIX_REUSE=0`) for one
release.

**Scope corrections from the code read (2026-09-18) — this task is bigger than
written:**

- **Reuse only on PURE EXTENSION (`lcp == prior_len`).** The text path's
  decision (`ar.rs:~3290`) degrades to checkpoint-resume or cold reset for
  `lcp < prior_len`, because **DeltaNet recurrent state cannot be partially
  retained the way KV can** — it is restored from a `prefill_checkpoints`
  snapshot, and the VL reset currently frees that ring. Accepting only pure
  extension keeps the recurrent state exactly correct by construction (it is
  the state after consuming precisely the retained tokens) and needs no
  checkpoint machinery. It also covers the agent case, which is append-only.
  Anything else must fall back to the full reset.

- **The prefix will NOT match unless assistant turns are spliced from cached
  token ids.** `continuation_suffix`'s doc states the reason: re-encoding a
  decoded reply is "a detokenise/retokenise round trip that is not guaranteed
  to be the identity". The VL framing added by the multi-turn fix re-renders
  the whole conversation from `messages` every turn, so a re-encoded assistant
  turn can differ from what generation actually produced, truncating `lcp`
  before the new content and defeating reuse. The text path already solves
  this: `asst_turn_fingerprint` + the assistant-turn cache splice the EXACT
  generated ids back into rendered history (`ar.rs:~3150`). VL framing must go
  through the same cache, or reuse will appear to work in tests using
  synthetic histories and rarely hit in production.

  This is the real dependency, and it means Task 5 is not "delete the reset" —
  it is "give VL the text path's continuation machinery". Budget accordingly,
  and prefer refactoring that machinery into something both paths call over
  reimplementing it in `vision.rs`.

- **Zero cursor arithmetic is needed for the pure-extension case.** Both turns
  frame from 0 and share a prefix, so the absolute positions of shared tokens
  are identical; the existing `MropeCtx` (built over the whole prompt, base 0)
  can simply be indexed from the resume point. `cursor_at_token` stays valuable
  as a cross-check and for any future partial-prefix work, but the first
  implementation does not need `base > 0` at all — which is why the guard can
  stay until then.

**Done when:** an image request repeated byte-identically reports `cached > 0`
and TTFT drops to roughly decode-start latency; extending a conversation by one
turn prefills only the new tokens; Task 0 fixtures still pass.

**Risk:** highest in the plan. The reset is load-bearing today. The kill switch
and Task 0's oracle are the mitigations; do not land this in the same PR as
Task 3.

---

## Task 6: DFlash on the VL path

**Goal:** Image-turn decode from ~32 to ~140 tok/s.

**Files:** `crates/hipfire-generate/src/vision.rs`,
`crates/hipfire-arch-qwen35/src/dflash_spec.rs`.

**Do:** `seed_target_hidden_from_prompt` cannot be reused — it re-prefills from
token ids. Add a seeding entry point that takes hidden states *already captured*
during the VL prefill: pass `Some(hidden_rb)` to the Task 2 batched call (the
parameter already exists), then seed the speculator from those rows instead of
re-running the target. Then route `generate_vl`'s decode through
`spec.step` like the text path, keeping the AR loop as fallback when
`m.speculator` is `None`.

**Done when:** image turns report `dflash: true` in `timings` with decode ≥ 100
tok/s, and greedy output is byte-identical to the AR path (DFlash verification
is exact — only τ changes).

**Note:** deliberately last. It depends on Task 2's `hidden_rb` plumbing and is
worth far less than Tasks 3/5 for an agent workload.

---

## Sequencing and PR boundaries

```
Task 0 (oracle, blocking)
   ├── Task 1 (cursor)        ─┐
   └── Task 2 (batch params)   │
          └── Task 3 (batched VL prefill)   ← PR 1: the 19×, no behaviour risk
                 └── Task 4 (prefix match)
                        └── Task 5 (drop reset)  ← PR 2: the re-prefill win
                               └── Task 6 (DFlash)  ← PR 3
```

Task 3 ships alone first: it is mechanical, verifiable against a measured
baseline, and cannot produce silently wrong output. Tasks 1+4+5 ship together
behind the kill switch, since a cursor without reuse is inert and reuse without
a cursor is wrong.

## Validation

Per PR, on gfx1201: the baseline table above re-measured; `cargo test -p` for
each touched crate; `scripts/leanup-ratchets.sh`; `scripts/fmt-changed.sh`.
Task 0 fixtures run in CI (CPU-only, no GPU needed). Correctness gate for Tasks
3 and 5 is token-identical greedy output against the pre-change binary on a
fixed conversation set, not eyeballed answers.

## What reuse does and does not buy (measured, gfx1201 / 32 GB)

Capacity is not the constraint. Model + sidecar + scratch resident is 21.42 GiB
of 31.86 GiB; KV costs **21.7 KiB/token** under `fwht3`, so the full 262 144-token
`max_seq` is ≈5.4 GiB and the whole context fits at ≈26.8 GiB with room to spare.

Reuse after Task 5 holds only under four conditions, all of which belong in the
operator docs alongside the feature:

1. **One conversation at a time.** hipfire keeps a single resident sequence.
   Two interleaved sessions each miss the other's prefix and force a full
   re-prefill. Cross-request prefix sharing is a paged-KV design (SGLang's
   radix tree, vLLM's block cache); it is not reachable from a single
   contiguous KV without a KV-layer rewrite, and is explicitly out of scope.
2. **`serve.idle_timeout` must be `0`.** The default 300 s unloads the model and
   takes the KV with it, so the next turn pays reload *and* re-prefill.
3. **Append-only history.** Editing or compacting earlier turns invalidates the
   prefix from the edit point — worth stating loudly, because context
   compaction is exactly what a long-running agent does.
4. **Eviction off**, or retained KV can be dropped out from under the match.

## Out of scope

Multi-image per request (the gateway still rejects it), audio chunks, the
experimental multi-slot engine (it refuses images and drafters outright), and
PFlash prompt compression (`speculation.prefill.*`, legacy, off, 32 k threshold
— it reduces token count rather than raising throughput).
