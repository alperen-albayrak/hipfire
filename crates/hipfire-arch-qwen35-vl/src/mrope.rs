// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 3D mrope position construction for Qwen3.5-VL.
//!
//! Mirrors `get_rope_index` / `get_vision_position_ids` in
//! `transformers/models/qwen3_5/modeling_qwen3_5.py`. Text tokens take the
//! same value on all three axes (which makes 3D mrope identical to 1D RoPE
//! for pure text); image tokens take their (t, h, w) grid coordinate.
//!
//! Pure CPU, no GPU, no I/O — this is the unit under test in
//! `tests/mrope_positions.rs`.

/// Default `mrope_section` for the Qwen3.5 family: 11 T + 11 H + 10 W = 32,
/// exactly the rotary-pair count (head_dim 256 * partial_rotary_factor 0.25
/// = 64 rotary dims = 32 pairs).
pub const DEFAULT_MROPE_SECTION: [usize; 3] = [11, 11, 10];

/// One contiguous run of visual tokens in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    /// Index of the first visual token.
    pub start: usize,
    /// Visual token count, POST-merge (== (grid_h/merge) * (grid_w/merge)).
    pub len: usize,
    /// PRE-merge grid height, as reported by the vision tower.
    pub grid_h: usize,
    /// PRE-merge grid width.
    pub grid_w: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MropePositions {
    /// Per-token (t, h, w).
    pub positions: Vec<[i32; 3]>,
    /// `positions.max() + 1 - n_tokens`. Added to the running sequence
    /// length to get decode-step positions.
    pub rope_delta: i32,
}

/// Which position axis frequency index `d` reads: 0 = T, 1 = H, 2 = W.
///
/// HF `apply_interleaved_mrope` starts every frequency on T, then overwrites
/// `slice(1, 3*section[1], 3)` with H and `slice(2, 3*section[2], 3)` with W,
/// producing `[THWTHW...TT]`. A chunked `[TTT...HHH...WWW]` reading is wrong.
pub fn mrope_axis_for_freq(d: usize, section: [usize; 3]) -> usize {
    match d % 3 {
        1 if d < 3 * section[1] => 1,
        2 if d < 3 * section[2] => 2,
        _ => 0,
    }
}

/// Build per-token (t, h, w) positions for a prompt containing `spans`
/// image runs. `spans` must be sorted by `start` and non-overlapping.
pub fn build_mrope_positions(
    n_tokens: usize,
    spans: &[ImageSpan],
    spatial_merge_size: usize,
) -> MropePositions {
    assert!(
        spatial_merge_size > 0,
        "spatial_merge_size must be positive"
    );
    let mut positions = Vec::with_capacity(n_tokens);
    let mut cursor: i32 = 0;
    let mut tok = 0usize;

    for span in spans {
        debug_assert!(span.start >= tok, "image spans must be sorted and disjoint");
        // Text run before this image.
        while tok < span.start && tok < n_tokens {
            positions.push([cursor; 3]);
            cursor += 1;
            tok += 1;
        }
        // Image run: t constant, h slowest, w fastest, over the merged grid.
        let lh = span.grid_h / spatial_merge_size;
        let lw = span.grid_w / spatial_merge_size;
        for hh in 0..lh {
            for ww in 0..lw {
                positions.push([cursor, cursor + hh as i32, cursor + ww as i32]);
            }
        }
        tok += lh * lw;
        // Advance by ONE GRID DIMENSION, not by the token count.
        cursor += (span.grid_h.max(span.grid_w) / spatial_merge_size) as i32;
    }

    // Trailing text run.
    while tok < n_tokens {
        positions.push([cursor; 3]);
        cursor += 1;
        tok += 1;
    }

    let max_pos = positions
        .iter()
        .flat_map(|p| p.iter().copied())
        .max()
        .unwrap_or(0);
    let rope_delta = max_pos + 1 - n_tokens as i32;

    MropePositions {
        positions,
        rope_delta,
    }
}

/// Position cursor after the first `upto` TOKENS of a prompt.
///
/// Token index and position are two coordinate systems: a text token consumes
/// one of each, but an image span consumes `len` token slots while advancing
/// the cursor by only one grid dimension (see [`build_mrope_positions`]). This
/// converts the former to the latter, so a prefill that RESUMES at token
/// `upto` can be positioned correctly.
///
/// Deliberately a pure function of the request rather than state carried on
/// the conversation. The full prompt — image spans included — is rebuilt on
/// every VL request, so the cursor is always derivable; storing it would add a
/// field that 76 `conversation_tokens.clear()` sites would each have to
/// remember to reset, and a missed one mis-positions every token after the
/// image with no error. This mirrors llama.cpp's `pos_from_tokens()`.
///
/// `spans` must be sorted by `start` and non-overlapping, as
/// [`build_mrope_positions`] requires.
pub fn cursor_at_token(upto: usize, spans: &[ImageSpan], spatial_merge_size: usize) -> i32 {
    assert!(
        spatial_merge_size > 0,
        "spatial_merge_size must be positive"
    );
    let mut cursor: i32 = 0;
    let mut tok = 0usize;

    for span in spans {
        if span.start >= upto {
            break;
        }
        // Text run before this image.
        while tok < span.start {
            cursor += 1;
            tok += 1;
        }
        // A partially-consumed image span cannot be resumed: the cursor is
        // only defined at span boundaries, so a prefix match must never cut
        // one in half. Callers truncate to the span start instead.
        debug_assert!(
            span.start + span.len <= upto,
            "cursor_at_token({upto}) splits an image span at {}..{}",
            span.start,
            span.start + span.len,
        );
        cursor += (span.grid_h.max(span.grid_w) / spatial_merge_size) as i32;
        tok += span.len;
    }

    // Trailing text run.
    while tok < upto {
        cursor += 1;
        tok += 1;
    }
    cursor
}

/// One image span in a framed conversation, tagged with the identity of the
/// image that produced it.
///
/// `content_id` must cover everything that changes the resulting visual
/// tokens, not just the image bytes: preprocessing config, resolved grid dims,
/// `spatial_merge_size`, and the vision sidecar's identity. Hashing bytes
/// alone can false-hit across a config change and splice embeddings that do
/// not match the retained KV. vLLM's `MultiModalHasher` folds `model_id` and
/// `hash_factors` in for exactly this reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaggedSpan {
    pub span: ImageSpan,
    pub content_id: u64,
}

/// Longest common prefix of two framed conversations, in TOKENS, safe to reuse
/// as KV.
///
/// A plain token walk is not merely insufficient here, it is wrong: image
/// positions are `image_pad` repeated, so two DIFFERENT images of the same
/// grid size produce byte-identical token runs and a token walk matches them.
/// This walks tokens but, at a span start, requires the `content_id` and token
/// length to agree before accepting the span — llama.cpp's `get_common_prefix`
/// media arm.
///
/// The result never splits a span: a mismatched image truncates the prefix at
/// its START, so the returned length is always a boundary at which
/// [`cursor_at_token`] is defined.
pub fn vl_common_prefix(
    a_tokens: &[u32],
    a_spans: &[TaggedSpan],
    b_tokens: &[u32],
    b_spans: &[TaggedSpan],
) -> usize {
    let max_idx = a_tokens.len().min(b_tokens.len());
    let span_at = |spans: &[TaggedSpan], i: usize| -> Option<TaggedSpan> {
        spans.iter().find(|t| t.span.start == i).copied()
    };

    let mut i = 0usize;
    while i < max_idx {
        match (span_at(a_spans, i), span_at(b_spans, i)) {
            (Some(x), Some(y)) => {
                // Same image, same extent -> the whole span is reusable.
                if x.content_id == y.content_id && x.span.len == y.span.len {
                    // A span running past the shorter prompt cannot be
                    // accepted: its KV is not fully present on both sides.
                    if i + x.span.len > max_idx {
                        return i;
                    }
                    i += x.span.len;
                    continue;
                }
                // Different image: truncate at the span start, never inside.
                return i;
            }
            // A span on one side and plain text on the other: divergence.
            (Some(_), None) | (None, Some(_)) => return i,
            (None, None) => {
                if a_tokens[i] != b_tokens[i] {
                    return i;
                }
                i += 1;
            }
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cursor reached after a prompt: the position the NEXT token would take.
    /// `max_pos + 1`, which `rope_delta` encodes as `+ n_tokens`.
    fn next_cursor(p: &MropePositions, n_tokens: usize) -> i32 {
        p.rope_delta + n_tokens as i32
    }

    fn span(start: usize, grid_h: usize, grid_w: usize, merge: usize) -> ImageSpan {
        ImageSpan {
            start,
            len: (grid_h / merge) * (grid_w / merge),
            grid_h,
            grid_w,
        }
    }

    /// A text-only run is positionally identical to 1-D RoPE: every axis holds
    /// the same value, so `mrope_axis_for_freq` cannot change what any
    /// frequency reads. This is the invariant that lets a conversation whose
    /// prefix was built by the TEXT path be reused by the VL path — without
    /// it, retained KV would have to be discarded on the first image turn.
    #[test]
    fn text_only_is_positionally_identical_to_1d_rope() {
        let p = build_mrope_positions(8, &[], 2);
        for (i, pos) in p.positions.iter().enumerate() {
            assert_eq!(*pos, [i as i32; 3], "text token {i} must be uniform");
        }
        assert_eq!(p.rope_delta, 0, "text-only must not skew the cursor");
        assert_eq!(next_cursor(&p, 8), 8);
    }

    /// An image advances the cursor by ONE GRID DIMENSION, not by its token
    /// count -- which is why position and token index diverge, and why the
    /// cursor cannot be recovered from `seq_pos` alone.
    #[test]
    fn image_advances_cursor_by_grid_dimension() {
        let merge = 2;
        let s = span(1, 16, 16, merge); // 64 post-merge tokens
        let n = 1 + s.len + 1;
        let p = build_mrope_positions(n, std::slice::from_ref(&s), merge);

        assert_eq!(p.positions[0], [0; 3], "leading text token");
        // 8x8 merged grid: t constant, h slowest, w fastest.
        assert_eq!(p.positions[1], [1, 1, 1], "first visual token");
        assert_eq!(p.positions[2], [1, 1, 2], "w advances fastest");
        assert_eq!(p.positions[1 + 8], [1, 2, 1], "h advances every row");

        // Cursor advanced by max(16,16)/2 = 8, NOT by the 64 tokens consumed.
        assert_eq!(p.positions[1 + s.len], [9; 3], "trailing text resumes at 9");
        assert_eq!(
            next_cursor(&p, n),
            10,
            "66 tokens but only 10 positions consumed"
        );
    }

    /// THE invariant Tasks 1/5 rest on: framing a whole conversation from 0
    /// must equal framing the prefix, then continuing the suffix at the
    /// prefix's next cursor. If this holds, retaining the prefix KV and
    /// prefilling only the suffix is position-exact -- which is the whole
    /// basis for dropping the force-reset.
    #[test]
    fn concatenation_equals_continuation() {
        let merge = 2;
        for (prefix_len, img_grid, suffix_len) in [
            (5usize, (16, 16), 7usize),
            (1, (8, 12), 3),
            (40, (32, 16), 11),
        ] {
            let img = span(prefix_len, img_grid.0, img_grid.1, merge);
            let whole_n = prefix_len + img.len + suffix_len;

            // One-shot: the whole conversation framed from scratch.
            let whole = build_mrope_positions(whole_n, std::slice::from_ref(&img), merge);

            // Incremental: prefix (text) framed, then the image+suffix framed
            // separately and shifted to the prefix's next cursor.
            let head = build_mrope_positions(prefix_len, &[], merge);
            let base = next_cursor(&head, prefix_len);
            let tail_span = span(0, img_grid.0, img_grid.1, merge);
            let tail_n = img.len + suffix_len;
            let tail = build_mrope_positions(tail_n, std::slice::from_ref(&tail_span), merge);

            let mut stitched: Vec<[i32; 3]> = head.positions.clone();
            stitched.extend(
                tail.positions
                    .iter()
                    .map(|p| [p[0] + base, p[1] + base, p[2] + base]),
            );

            assert_eq!(
                stitched, whole.positions,
                "prefix_len={prefix_len} grid={img_grid:?} suffix_len={suffix_len}: \
                 continuation must reproduce one-shot framing exactly"
            );

            // And the cursors must agree, or turn N+2 drifts.
            assert_eq!(
                next_cursor(&whole, whole_n),
                base + next_cursor(&tail, tail_n),
                "cursor must compose across the seam"
            );
        }
    }

    /// `cursor_at_token` must agree with `build_mrope_positions` at the end of
    /// the prompt, and at every span boundary in between. These are two
    /// independent walks of the same structure; if they ever disagree, a
    /// resumed prefill positions its first token wrong and everything after
    /// the image is skewed -- silently.
    #[test]
    fn cursor_agrees_with_builder_at_every_boundary() {
        let merge = 2;
        for (pre, (gh, gw), mid, (gh2, gw2), post) in [
            (
                5usize,
                (16usize, 16usize),
                7usize,
                (8usize, 12usize),
                3usize,
            ),
            (0, (8, 8), 1, (16, 32), 11),
            (40, (32, 16), 0, (8, 8), 9),
        ] {
            let a = span(pre, gh, gw, merge);
            let b_start = pre + a.len + mid;
            let b = span(b_start, gh2, gw2, merge);
            let n = b_start + b.len + post;
            let spans = [a, b];

            let built = build_mrope_positions(n, &spans, merge);

            // End of prompt: the cursor is max_pos + 1, which rope_delta
            // encodes relative to the token count.
            assert_eq!(
                cursor_at_token(n, &spans, merge),
                next_cursor(&built, n),
                "end-of-prompt cursor must match rope_delta",
            );

            // Every boundary: the cursor equals the position the next token
            // takes, which the builder wrote for that token.
            for boundary in [pre, pre + a.len, b_start, b_start + b.len] {
                if boundary >= n {
                    continue;
                }
                assert_eq!(
                    cursor_at_token(boundary, &spans, merge),
                    built.positions[boundary][0],
                    "cursor at token {boundary} must equal that token's t-axis position",
                );
            }
        }
    }

    /// Resuming at a boundary reproduces one-shot framing -- the same property
    /// `concatenation_equals_continuation` proves for positions, now driven by
    /// the cursor a resumed prefill would actually compute.
    #[test]
    fn cursor_drives_exact_resumption() {
        let merge = 2;
        let img = span(6, 16, 16, merge);
        let n = 6 + img.len + 9;
        let spans = [img];
        let whole = build_mrope_positions(n, &spans, merge);

        // Resume from just before the image: prefix is pure text.
        let resume_at = 6usize;
        let base = cursor_at_token(resume_at, &spans, merge);
        assert_eq!(base, 6, "six text tokens consume six positions");

        let tail_span = span(0, 16, 16, merge);
        let tail = build_mrope_positions(n - resume_at, std::slice::from_ref(&tail_span), merge);
        let stitched: Vec<[i32; 3]> = tail
            .positions
            .iter()
            .map(|p| [p[0] + base, p[1] + base, p[2] + base])
            .collect();

        assert_eq!(
            stitched,
            whole.positions[resume_at..],
            "a prefill resumed at the cursor must match one-shot framing",
        );
    }

    fn tagged(start: usize, gh: usize, gw: usize, merge: usize, id: u64) -> TaggedSpan {
        TaggedSpan {
            span: span(start, gh, gw, merge),
            content_id: id,
        }
    }

    /// Two DIFFERENT images produce byte-identical token runs (image_pad
    /// repeated), so a token-only walk matches them and would reuse KV that
    /// belongs to another picture. This is the correctness case, not an
    /// optimisation: the prefix must truncate at the span START.
    #[test]
    fn different_image_same_shape_does_not_match() {
        let merge = 2;
        let pad = 9999u32;
        let a_img = tagged(3, 16, 16, merge, 0xAAAA);
        let b_img = tagged(3, 16, 16, merge, 0xBBBB); // same shape, other image
        let mut toks: Vec<u32> = vec![1, 2, 3];
        toks.extend(std::iter::repeat(pad).take(a_img.span.len));
        toks.extend([7, 8]);

        // Token runs are identical on both sides -- that is the trap.
        assert_eq!(a_img.span.len, b_img.span.len);
        let cut = vl_common_prefix(&toks, &[a_img], &toks, &[b_img]);
        assert_eq!(
            cut, 3,
            "must truncate at the image start, not inside or past it"
        );
    }

    /// The same image in the same place is reusable, and the walk continues
    /// past it into the following text.
    #[test]
    fn same_image_matches_and_walk_continues() {
        let merge = 2;
        let pad = 9999u32;
        let img = tagged(3, 16, 16, merge, 0xAAAA);
        let mut a: Vec<u32> = vec![1, 2, 3];
        a.extend(std::iter::repeat(pad).take(img.span.len));
        a.extend([7, 8, 9]);
        let mut b = a.clone();
        // Diverge only AFTER the image.
        let last = b.len() - 1;
        b[last] = 42;

        let cut = vl_common_prefix(&a, &[img], &b, &[img]);
        assert_eq!(
            cut,
            a.len() - 1,
            "identical image must be traversed, divergence found in trailing text",
        );
        // The cut is a boundary where the cursor is defined.
        let _ = cursor_at_token(cut, &[img.span], merge);
    }

    /// An image against plain text at the same index is divergence, not a
    /// token comparison -- `image_pad` could otherwise coincide with a real id.
    #[test]
    fn image_versus_text_diverges_at_the_span() {
        let merge = 2;
        let pad = 9999u32;
        let img = tagged(2, 8, 8, merge, 0xAAAA);
        let mut a: Vec<u32> = vec![1, 2];
        a.extend(std::iter::repeat(pad).take(img.span.len));
        let b: Vec<u32> = std::iter::repeat(pad).take(a.len()).collect();
        let b = [vec![1u32, 2], b[2..].to_vec()].concat();

        let cut = vl_common_prefix(&a, &[img], &b, &[]);
        assert_eq!(cut, 2, "text side must not absorb an image span");
    }

    /// A span that runs past the shorter conversation is not reusable: its KV
    /// is not fully present on both sides.
    #[test]
    fn span_overrunning_the_shorter_side_is_rejected() {
        let merge = 2;
        let pad = 9999u32;
        let img = tagged(2, 16, 16, merge, 0xAAAA);
        let mut a: Vec<u32> = vec![1, 2];
        a.extend(std::iter::repeat(pad).take(img.span.len));
        let b: Vec<u32> = a[..a.len() - 4].to_vec(); // truncated mid-image

        let cut = vl_common_prefix(&a, &[img], &b, &[img]);
        assert_eq!(cut, 2, "must not accept a partially present span");
    }

    /// The frequency-to-axis map is interleaved `[THWTHW...TT]`, not chunked.
    /// Pinned because a chunked reading is a silent, plausible-looking bug.
    #[test]
    fn axis_map_is_interleaved_not_chunked() {
        let section = [11, 11, 10];
        assert_eq!(mrope_axis_for_freq(0, section), 0, "d=0 -> T");
        assert_eq!(mrope_axis_for_freq(1, section), 1, "d=1 -> H");
        assert_eq!(mrope_axis_for_freq(2, section), 2, "d=2 -> W");
        assert_eq!(mrope_axis_for_freq(3, section), 0, "d=3 -> T again");
        // Past 3*section[i] the axis falls back to T -- the trailing `TT`.
        assert_eq!(mrope_axis_for_freq(3 * 10 + 2, section), 0, "W exhausted");
    }
}
