// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched sibling of `test_mrope_rope_parity`.
//!
//! That example proves the PER-TOKEN mrope kernel degenerates exactly to 1D
//! RoPE when t == h == w. This one proves the same for the BATCHED pair, which
//! is the claim batched VL prefill rests on:
//!
//!   1. `rope_mrope_halfsplit_f32_batched` with t == h == w must be
//!      bit-identical to `rope_partial_interleaved_f32_batched`. Without this,
//!      a conversation prefix written by the TEXT path cannot be reused by a
//!      VL turn — the stored K would disagree — and killing the VL
//!      force-reset would silently corrupt reused context.
//!
//!   2. The same holds with a non-zero `pos_offset`. Nothing in-tree calls the
//!      batched mrope kernel today, so its offset argument — the cross-turn
//!      position cursor — is entirely unexercised.
//!
//! Run: cargo run --release -p rdna-compute --example test_mrope_rope_parity_batched --features lab,deltanet

use rdna_compute::Gpu;

fn lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) & 0x7fff) as f32 / 32_768.0 - 0.5
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    // Qwen3.8-27B full-attention geometry.
    let (nhq, nhk, hd, n_rot) = (24usize, 4usize, 256usize, 64usize);
    let freq_base = 1_000_000.0f32;
    let section = [11usize, 11, 10];
    let n = 37usize; // deliberately not a round number / wave multiple

    let mut failures = 0usize;

    for pos_offset in [0i32, 713, 4096] {
        let qd = lcg(0xa1, n * nhq * hd);
        let kd = lcg(0xb2, n * nhk * hd);
        // Sequential token positions, as a real prefill would carry.
        let positions: Vec<i32> = (0..n as i32).map(|i| i + 5).collect();

        // ── Reference: batched 1D kernel ────────────────────────────────
        let q1 = gpu.upload_f32(&qd, &[n * nhq * hd]).unwrap();
        let k1 = gpu.upload_f32(&kd, &[n * nhk * hd]).unwrap();
        // i32 bits in an F32 tensor — the dtype-cosmetic pattern the prefill
        // batch buffers use; the kernels cast to `const int*`.
        let p1 = gpu.upload_f32(&vec![0.0f32; n], &[n]).unwrap();
        let p1_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&p1.buf, &p1_bytes).unwrap();
        gpu.rope_partial_interleaved_f32_batched(
            &q1, &k1, &p1, nhq, nhk, hd, n_rot, freq_base, n, pos_offset,
        )
        .unwrap();

        // ── Candidate: batched mrope with t == h == w ───────────────────
        let q2 = gpu.upload_f32(&qd, &[n * nhq * hd]).unwrap();
        let k2 = gpu.upload_f32(&kd, &[n * nhk * hd]).unwrap();
        let p2_bytes: Vec<u8> = positions
            .iter()
            .flat_map(|v| [*v, *v, *v])
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let p2 = gpu.hip.malloc(p2_bytes.len()).unwrap();
        gpu.hip.memcpy_htod(&p2, &p2_bytes).unwrap();
        gpu.rope_mrope_halfsplit_f32_batched(
            &q2, &k2, &p2, nhq, nhk, hd, n_rot, freq_base, n, pos_offset, section,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let (a, b) = (
            gpu.download_f32(&q1).unwrap(),
            gpu.download_f32(&q2).unwrap(),
        );
        let (c, d) = (
            gpu.download_f32(&k1).unwrap(),
            gpu.download_f32(&k2).unwrap(),
        );
        let dq = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dk = c
            .iter()
            .zip(&d)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let ok = dq == 0.0 && dk == 0.0;
        println!(
            "pos_offset={pos_offset:<5} n={n} max|dq|={dq:.3e} max|dk|={dk:.3e}  {}",
            if ok { "OK" } else { "MISMATCH" }
        );
        if !ok {
            failures += 1;
        }
    }

    assert_eq!(
        failures, 0,
        "batched mrope with t==h==w must be BIT-IDENTICAL to batched 1D rope; \
         a mismatch means a text-written prefix cannot be reused by a VL turn"
    );
    println!("PASS: batched mrope degenerates exactly to 1D RoPE, offset included");
}
