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
