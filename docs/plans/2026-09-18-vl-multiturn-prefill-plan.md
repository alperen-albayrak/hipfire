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

**Design, derived from the code (2026-09-18):**

`DflashSpeculator::prefill` is already two separable phases:

1. `seed_target_hidden_from_prompt_abortable(...)` — runs the TARGET forward
   over `prefill_tokens`, capturing hidden states into `self.df.hidden_rb`.
   **This is the only token-based phase, and the only one VL cannot reuse**,
   because image positions are embeddings with no token spelling.
2. `scatter_hidden_block_to_interleaved(gpu, &self.df.hidden_rb,
   &self.df.draft_scratch.target_hidden, ...)` — primes the drafter purely
   from those captured rows. **Provenance-agnostic**, so it works unchanged
   for a VL prefill.

So VL substitutes its own phase 1 and reuses phase 2 verbatim. Two additive
trait methods, both with defaults so no other `Speculator` impl changes:

```rust
fn hidden_rb_mut(&mut self) -> Option<&mut HiddenStateRingBuffer> { None }
fn prime_from_hidden(&mut self, gpu: &mut Gpu, prompt_len: usize)
    -> Result<(), String> { Err("unsupported".into()) }
```

`generate_vl` then borrows the ring buffer out of the speculator, passes it as
the `hidden_rb` argument of the Task 3 batched call (**the parameter already
exists and is currently `None`**), and calls `prime_from_hidden` afterwards.

### Prior art — how the references do vision + speculative decode

Checked 2026-09-18 against llama.cpp, vLLM and SGLang. All three converge on
the same shape, and none of them feeds visual embeddings to the drafter.

- **llama.cpp (EAGLE3, `common/speculative.cpp:577`)** skips any batch that
  carries embeddings outright:
  `if (batch_in.token == nullptr || batch_in.embd != nullptr) { return true; }`
  Since `mtmd` submits images exactly that way (`decode_embd_batch`), the
  drafter simply never ingests image rows. No error, no disable — speculation
  continues over the text with a hole where the image was. There is no
  mtmd/spec incompatibility check in the server.
- **vLLM (`v1/spec_decode/llm_base_proposer.py`)** warns and degrades:
  "does not fully support multimodal models yet. **Proceeding with text-only
  speculative decoding.**" It also hard-fails if the DRAFT itself wants
  M-RoPE (`_raise_if_mrope`), and comments that the draft's M-RoPE setting —
  not the target's — is what counts, because "draft models may be text-only
  even if target is multimodal".
- **SGLang** tolerates rather than skips: its EAGLE draft embedding "clamps
  unconditionally (to tolerate multimodal pad sentinels)", and its DSpark
  verify path excludes requests carrying `input_embeds` or
  `multimodal_inputs` from position-bounding because "their visible token IDs
  may not track cache positions".

**What this means for hipfire.** The draft here is text-only 1-D —
`dflash_spec.rs` contains zero mrope references — which is exactly the
configuration all three references assume, so no M-RoPE work is needed on the
draft side.

hipfire is actually better placed than llama.cpp for the seeding: DFlash is
seeded from TARGET hidden states, and the target computes real hidden states
for image rows during its own VL prefill. So the rows exist and there is no
hole to skip — unlike EAGLE3, which never sees them because it works off the
batch.

The residual issue is the draft's own token embedding at image positions,
where the id is `image_pad` — a real vocab id with no meaning. Follow SGLang
and tolerate it (clamp/accept) rather than special-case it: acceptance simply
degrades for the few tokens following an image and recovers. Do NOT expect
image-adjacent tau to match text; none of the references achieve that, and a
low tau there is expected behaviour, not a bug to chase.

### CORRECTION (2026-09-18): this task is correctness-sensitive, not throughput-only

An earlier note here claimed the worst case for Task 6 was "no speedup", on the
grounds that DFlash verify is exact. **That was wrong.** Exactness only means
the drafter is checked against *the target*; if the target itself rotates at
the wrong phase, verification faithfully reproduces a wrong answer.

`MropeCtx::pos3` (`qwen35/config.rs:773`) falls off the end of the prompt into:

```rust
None => [pos as i32 + self.rope_delta; 3],   // decode steps
```

So a VL decode step rotates at **`pos + rope_delta`**, not `pos`. Observed
`rope_delta = -56` for a 64x64 image (64 visual tokens collapsing to an 8-wide
grid advance). `SpecTarget`'s forwards are plain 1-D at `pos` — `spec.rs`
contains zero mrope references — so routing VL decode through `spec.step` as-is
would rotate every decode token 56 positions off. Wrong logits, wrong output.

**But the fix is small**, because decode-step mrope is UNIFORM (`[p; 3]`, same
on all three axes) and therefore numerically plain 1-D RoPE at a shifted
position. What the spec path needs is a scalar rope-phase bias, not 3-axis
mrope.

hipfire already has that mechanism: the batched rope call takes
`kv_cache.compact_offset` as a rope-only offset while `pbs.positions` stays
physical for the KV write. `rope_delta` is the same shape of thing — a phase
bias that must never reach slot indices.

**Revised order for this task:**

1. Thread a rope-phase bias (default 0, so text is byte-identical) through
   `SpecTarget`'s advance/verify forwards.
2. **GPU parity harness FIRST**: AR-decoded VL output vs spec-decoded,
   token-for-token at temperature 0. The channel-invariance suite covers
   formatting; this covers POSITIONS, which is the new risk and the one that
   produces plausible-looking wrong text.
3. Only then wire the decode loop.

Do not wire the loop before (1): without the bias the change is actively
wrong, not merely ineffective.

### Decode-loop implementation map (surveyed 2026-09-18)

Everything below is located, so the port starts without rediscovery.

**The AR loop to mirror** lives in `generate_vl`, roughly lines 1610-1953 of
`vision.rs`. Its hot forward is the single `forward_scratch_mrope` at ~1661.
The think force-close is nested INSIDE the loop (~1817, 24-space indent); the
ChatML `\n` boundary-sync forward is in the EPILOGUE, after the loop ends
(~1960). Only the hot forward is replaced by `spec.step`; the other two stay
per-token and occasional.

**A working VL + speculator template already exists in the same file**:
`decode_vl_dots_ocr_ngram` (~2550) and `run_dots_ocr_ngram_loop` (~2576),
the dots-ocr n-gram path. Its shape is the one to copy — take the bundle and
the speculator OUT of `m` (disjoint fields), run a dedicated loop that never
touches `m`, then restore both. Its guard comment is worth reading: the
prefill bindings must be released first so the branch can take `&mut m`.

**Getting a `SpecTarget` for qwen35** differs from dots-ocr, whose bundle IS
the target. Here: `Qwen35SlotGuard::take(&mut m.state, &m.model_path)` returns
an RAII guard; `.slot()` yields `&mut dyn SpecTarget`; `Drop` converts the slot
back into the bundle and restores it. `ModelSlot` already carries
`vision_config` / `vision_weights` so a VL bundle round-trips without loss.

**Seeding.** The dots-ocr loop primes with
`spec.prefill(cache_hit = true, empty suffix)`, which skips the target advance
and just argmaxes the live vision-conditioned state. That is enough for n-gram,
which keeps no hidden state — but NOT for DFlash, whose drafter needs the
prompt's hidden rows. For DFlash: pass the drafter's ring as `hidden_rb` to the
VL prefill (the parameter is already there, currently `None`, with the call
site noted), then `prime_from_hidden`.

**Set the bias before stepping**: `target.set_rope_phase_bias(rope_delta)` from
the request's `MropeCtx`, or every decode token rotates at the wrong phase.

### Prerequisite finding (2026-09-18): "byte-identical to AR" is the WRONG gate

Measured before building, on the TEXT path where both AR and DFlash already
work, via `HIPFIRE_SPECULATION=off` (no config mutation needed). Greedy,
temperature 0, fixed seed, `scripts/ar_spec_diff.py`:

| | AR vs spec |
|---|---|
| text-short, text-multi, both image cases | **identical** |
| text-prose | **DIVERGED** at char 169 — spec "two smaller whole numbers", AR "two smaller positive integers" |

Same token count, different words. So a VL gate asserting token-for-token
identity with AR would have failed for reasons unrelated to the change.

The cause looks benign and is worth recording: verify runs a BATCHED forward,
AR runs PER-TOKEN, and those two routes differ by ~7.5e-3 in logits (measured
independently by the rope-phase parity harness). At a near-tie that flips the
argmax, and the sequences part from there. AGENTS.md already notes
"draft-target argmax disagreement on prose tokens".

**Spec output IS deterministic**: identical across two runs and a server
restart, all five cases. That is what makes a regression gate possible.

**So the gate for the decode loop is:**

1. **Determinism** — VL-spec output reproducible across runs/restarts. Catches
   state leakage between requests, which is the failure a single run hides.
2. **Prefix agreement vs VL-AR** — divergence no worse than the text baseline.
   Text agreed for 169 chars before a near-tie; VL-spec diverging at token 1-2
   means a position or seeding bug, not a near-tie.
3. **Mechanism tests, already green** — rope-phase bit-identity
   (`test_spec_rope_phase_bias_parity`) and channel-delivery invariance
   (`vision.rs` tests).
4. **tau > 1** — the drafter must actually accept something, else the wiring is
   inert and passing tests prove nothing.

Do NOT tighten (2) into byte-identity. It would be measuring the batched-vs-
per-token float difference, not the correctness of this change.

**Gate it.** Land behind an env flag defaulting OFF (the repo gates
`HIPFIRE_JINJA_CHAT`, `HIPFIRE_DFLASH_VERIFY_PM4`, `vision_mode` the same way),
so the port is reviewable and measurable without changing default behaviour.

### STATUS 2026-09-18: wired, gated, and FAULTING — do not enable

`HIPFIRE_VL_DFLASH=1` faults the GPU on gfx1201:

    Memory access fault by GPU node-1 ... Reason: Page not present

Twice, reproducibly. The gate contained it both times: default-off meant only
the explicit opt-in was affected, and after restarting without it the whole
suite is byte-identical to the pre-change baseline on all five cases.

**The fault is in PREFILL, not in priming or decode.** The daemon log ends:

    [daemon/vl] mrope: span start=4 ... rope_delta=-56
      vision done: 64 tokens x 5120 dims
    Memory access fault by GPU node-1 ...

No `dflash prime` or `dflash decode` line is ever reached. So the failing step
is `forward_prefill_batch` with `hidden_rb = Some(drafter ring)` — the hidden
capture itself — on a prompt of only ~85 tokens.

A first hypothesis (missing `reset_upload_tracking` / `last_window` in
`prime_from_hidden`, which `prefill` does and `prime_from_hidden` omitted) was
a REAL gap and is fixed, but it was not this fault: priming never runs.

**Where to look next, not yet investigated:**

- `seed_target_hidden_from_prompt` passes `&mut self.df.hidden_rb` to
  `forward_prefill_batch` with the target as a **`ModelSlot`** — its own
  `weights` / `config` / `kv_cache` / `dn_state` / `scratch`. The VL path
  passes the **bundle's** scratch instead. If the ring's staging expects a
  scratch sized for hidden capture, the bundle's is the wrong one.
- `forward_prefill_batch`'s chunk sizing consults
  `hidden_rb.as_ref().map(|rb| rb.max_batch)` (prefill.rs ~493, ~1336), and
  the VL call passes `pbs_in: None` so the batch scratch is allocated
  internally. Whether that allocation accounts for hidden staging is unchecked.

**Do not brute-force this against the live server.** Each fault kills the
daemon and needs a restart. Isolate it in a GPU example (as
`test_spec_rope_phase_bias_parity` does) that runs one VL-shaped prefill with
`hidden_rb = Some`, so the failure is reproducible without taking the service
down.

**The remaining bulk is the decode loop, not the seeding.** `generate_vl` has a
bespoke AR decode loop carrying think-routing (`<think>`/`</think>` pairing and
force-close), the emit contract, abort polling, eviction and adaptive
downshift. Routing it through `spec.step` means re-expressing all of that on
the speculator's acceptance-window model. The risk here is NOT performance —
it is silently changing reasoning-channel splitting or stop handling, which
shows up as malformed output rather than an error. Budget for a reference test
comparing emitted channels token-for-token against the AR path before switching
the default, and keep the AR loop as the fallback when `m.speculator` is
`None`.

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
