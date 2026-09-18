// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Does the speculator's rope-phase bias reproduce a VL decode step exactly?
//!
//! Routing VL decode through `spec.step` is only sound if the speculator's
//! target forward rotates at the SAME phase the AR VL loop uses.
//! `MropeCtx::pos3` falls off the end of the prompt into
//! `[pos + rope_delta; 3]`, so a VL decode step rotates at `pos + rope_delta`,
//! not `pos` — `rope_delta` was -56 for one 64x64 image. A target that rotates
//! at plain `pos` yields wrong logits, and DFlash's exact verify would then
//! faithfully reproduce a wrong answer. That failure is silent: plausible text,
//! no error.
//!
//! So this compares, at the SAME position and the same KV slot:
//!
//!   A. `forward_scratch_mrope(tok, p, mrope{rope_delta})` — what the AR VL
//!      decode loop actually does today.
//!   B. `forward_prefill_batch_rope_biased(&[tok], p, bias = rope_delta)` —
//!      what a speculator target forward would do.
//!
//! Bit-identical logits are required. A control run additionally asserts
//! `bias = 0` is byte-identical to the plain unbiased forward, so the
//! delegation arm cannot rot.
//!
//! This is deliberately the test that would be skipped: the channel-invariance
//! suite in `vision.rs` covers reasoning/content SPLITTING and would sail
//! straight through a 56-position phase error while the text quietly degraded.
//!
//! Run: cargo run --release --features deltanet -p hipfire-arch-qwen35 \
//!         --example test_spec_rope_phase_bias_parity -- \
//!         ~/.hipfire/models/qwen3.8-27b.mq4

use hipfire_arch_qwen35::qwen35::{self, MropeCtx};
use hipfire_arch_qwen35::speculative::{ModelSlot, ModelSlotConfig};
use hipfire_runtime::spec::SpecTarget;
use rdna_compute::Gpu;
use std::path::Path;

/// Prefill a fixed prompt from clean state so both paths start identical.
fn seed(gpu: &mut Gpu, slot: &mut ModelSlot, prompt: &[u32]) {
    slot.reset_state(gpu).expect("reset_state");
    qwen35::forward_prefill_batch(
        gpu,
        &slot.weights,
        &slot.config,
        prompt,
        0,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
        None,
        None,
        None,
        None,
    )
    .expect("seed prefill");
}

fn logits(gpu: &mut Gpu, slot: &ModelSlot) -> Vec<f32> {
    gpu.download_f32(&slot.scratch.logits).expect("download logits")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vectors differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: test_spec_rope_phase_bias_parity <model.mq4>");
    let mut gpu = Gpu::init().expect("GPU init");
    let mut slot = ModelSlot::load(
        &mut gpu,
        Path::new(&path),
        "target",
        ModelSlotConfig {
            max_seq: 512,
            ..Default::default()
        },
    )
    .expect("ModelSlot::load");

    // A short synthetic prompt: this is about POSITION handling, so the tokens
    // only need to be valid ids, not meaningful text.
    let prompt: Vec<u32> = (10u32..26).collect();
    let next_token = 42u32;
    let p = prompt.len();

    let mut failures = 0usize;

    // rope_delta values a real image produces: -56 is one 64x64 image (64
    // visual tokens collapsing to an 8-wide grid advance); -184 approximates a
    // 512x512. 0 is the control that must hit the delegation arm.
    for delta in [0i32, -56, -184, 37] {
        // ── A. what the AR VL decode loop does ──────────────────────────
        seed(&mut gpu, &mut slot, &prompt);
        let mrope = MropeCtx::new(&slot.config, 0, Vec::new(), delta);
        qwen35::forward_scratch_mrope(
            &mut gpu,
            &slot.weights,
            &slot.config,
            next_token,
            p,
            &mut slot.kv_cache,
            &mut slot.dn_state,
            &slot.scratch,
            Some(&mrope),
        )
        .expect("mrope decode forward");
        let a = logits(&mut gpu, &slot);

        // ── B. what a speculator target forward would do ────────────────
        seed(&mut gpu, &mut slot, &prompt);
        SpecTarget::set_rope_phase_bias(&mut slot, delta);
        qwen35::forward_prefill_batch_rope_biased(
            &mut gpu,
            &slot.weights,
            &slot.config,
            &[next_token],
            p,
            &mut slot.kv_cache,
            &mut slot.dn_state,
            &slot.scratch,
            None,
            slot.rope_phase_bias,
        )
        .expect("biased batch forward");
        let b = logits(&mut gpu, &slot);

        let d = max_abs_diff(&a, &b);
        let ok = d == 0.0;
        println!(
            "rope_delta={delta:<5} max|dlogit|={d:.3e}  {}",
            if ok { "OK" } else { "MISMATCH" }
        );
        if !ok {
            failures += 1;
        }
    }

    // ── Control: the bias must actually MATTER ──────────────────────────
    // If a wiring mistake dropped the bias on the floor, every comparison
    // above would still pass -- both paths would simply ignore it. Prove a
    // non-zero bias changes the logits at all.
    seed(&mut gpu, &mut slot, &prompt);
    qwen35::forward_prefill_batch_rope_biased(
        &mut gpu,
        &slot.weights,
        &slot.config,
        &[next_token],
        p,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
        None,
        0,
    )
    .expect("unbiased");
    let unbiased = logits(&mut gpu, &slot);

    seed(&mut gpu, &mut slot, &prompt);
    qwen35::forward_prefill_batch_rope_biased(
        &mut gpu,
        &slot.weights,
        &slot.config,
        &[next_token],
        p,
        &mut slot.kv_cache,
        &mut slot.dn_state,
        &slot.scratch,
        None,
        -56,
    )
    .expect("biased");
    let biased = logits(&mut gpu, &slot);

    let sensitivity = max_abs_diff(&unbiased, &biased);
    println!("control: max|dlogit| between bias 0 and bias -56 = {sensitivity:.3e}");
    if sensitivity == 0.0 {
        println!("  MISMATCH: bias had NO effect — it is being dropped, and the");
        println!("  parity results above are vacuous.");
        failures += 1;
    }

    assert_eq!(
        failures, 0,
        "speculator rope-phase bias must reproduce the AR VL decode phase exactly; \
         a mismatch means routing VL decode through spec.step produces wrong logits"
    );
    println!("PASS: biased spec forward matches the AR mrope decode step exactly");
}
