// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Where does long-context DECODE attention lose its bandwidth?
//!
//! Measured end-to-end on cachy-01 (gfx1201, qwen3.8-27b.mq4, fwht3 K / Q8 V,
//! DFlash2) on 2026-09-21:
//!
//!   ~200-token context: 140 tok/s, tau ~3.3 => 23.8 ms/verify-cycle.
//!     Per cycle the GPU reads 14.59 GiB target weights + 1.13 GiB draft
//!     = 15.7 GiB => ~660 GB/s. Essentially roofline. Nothing wrong.
//!   124,345-token context: 7.8 tok/s, tau 2.03 => 260 ms/verify-cycle.
//!     Per cycle 14.59 + 1.13 + 2.96 GiB of KV = 18.7 GiB => ~72 GB/s.
//!     The memory model says that cycle should take 28 ms. It takes 260.
//!
//! tau accounts for only 1.6x of the 18x decode collapse. Attributing the
//! remaining ~232 ms to attention over the KV gives ~14.5 ms per
//! full-attention layer to read 185 MB — about **2% of peak bandwidth**.
//!
//! This bench isolates that one kernel so the 2% figure can be confirmed or
//! refuted without the 20-minute prefill in front of it, and separates two
//! candidate causes:
//!
//!   A. **Record alignment.** fwht3 K is 100 B/head (4 B cnorm + 96 B of
//!      3-bit codes) — not a power of two, not cache-line aligned, so every
//!      head's read straddles lines. Q8 V at 272 B/head has the same shape of
//!      problem. If this dominates, GB/s stays flat and low at every seq_len.
//!   B. **Partials sized for the ALLOCATION, not the request.**
//!      `llama_flash_partials_len` scales tiles by `max_seq` — 262144 in
//!      production — while a request may only span 124K. The `max_seq` vs
//!      `max_ctx_len` split in the launcher signature lets us vary exactly
//!      that: same work, different tile count. If this dominates, the tight
//!      arm beats the production arm at the same seq_len.
//!
//! Batch is the DFlash verify block (8), not a prefill batch — which is also
//! why this path can never reach the gfx1201 FA2 kernel: that one requires
//! `batch_size` in 64..=512.
//!
//! Run (GPU-locked):
//!   source scripts/gpu-lock.sh && gpu_acquire "longctx-decode-attn" \
//!     && ./target/release/examples/bench_longctx_decode_attn; \
//!     rc=$?; gpu_release; echo RUN_RC=$rc

use rdna_compute::{DType, Gpu};

/// Qwen3.8-27B full-attention geometry, read from the checkpoint header.
const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
/// DFlash2 draft block — the verify batch this path actually sees.
const BATCH: usize = 8;
/// Production allocation: `max_position_embeddings` for this model.
const PROD_MAX_SEQ: usize = 262_144;

/// asym3 K record: 4-byte f32 cnorm + head_dim*3/8 bytes of packed codes.
const fn k_bytes_per_head() -> usize {
    4 + (HEAD_DIM * 3) / 8
}
/// Q8_0 V: one 34-byte block per 32 elements.
const fn v_bytes_per_head() -> usize {
    (HEAD_DIM / 32) * 34
}

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
    let args: Vec<String> = std::env::args().collect();
    let argval = |k: &str, d: usize| {
        args.iter()
            .position(|a| a == k)
            .map(|i| args[i + 1].parse().unwrap())
            .unwrap_or(d)
    };
    let iters = argval("--iters", 20);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "bench_longctx_decode_attn: arch={} batch={BATCH} H{N_HEADS}/KV{N_KV_HEADS}/D{HEAD_DIM}",
        gpu.arch
    );
    eprintln!(
        "  asym3 K = {} B/head ({} B/pos), Q8 V = {} B/head ({} B/pos) => {} B/pos/layer",
        k_bytes_per_head(),
        k_bytes_per_head() * N_KV_HEADS,
        v_bytes_per_head(),
        v_bytes_per_head() * N_KV_HEADS,
        (k_bytes_per_head() + v_bytes_per_head()) * N_KV_HEADS
    );
    eprintln!();
    println!(
        "{:>9}  {:>6}  {:>6}  {:>9}  {:>9}  {:>11}",
        "seq_len", "batch", "tier", "ms/call", "GB/s", "us/token"
    );

    // FWHT-256 sign tables, as the fwht3 K-write used.
    let signs1 = gpu.upload_f32(&vec![1.0f32; 256], &[256]).unwrap();
    let signs2 = gpu.upload_f32(&vec![1.0f32; 256], &[256]).unwrap();

    // ── Experiment 1: does the kernel care about BYTES? ────────────────
    // fwht3 K is 400 B/pos, Q8 K is 1088 B/pos — 1.46x more total KV bytes
    // for identical work. If this kernel is bandwidth-bound, Q8 must be
    // ~1.46x slower. If the two take the SAME time, bytes are not the
    // limiter and fwht3's compression buys nothing here.
    //
    // ── Experiment 2: is it starved at the verify batch? ────────────────
    // The DFlash verify batch is 8. The tuned gfx1201 FA2 kernel requires
    // batch_size in 64..=512 — so if per-token cost falls sharply as batch
    // grows, the decode path is occupancy-starved at exactly the batch it
    // always runs at, and that is the same wall FA2's floor describes.
    let max_seq = PROD_MAX_SEQ;
    let k_bytes = max_seq * N_KV_HEADS * k_bytes_per_head();
    let v_bytes = max_seq * N_KV_HEADS * v_bytes_per_head();
    // Q8 K is 272 B/head, the same layout V already uses.
    let k8_bytes = max_seq * N_KV_HEADS * v_bytes_per_head();
    let k_cache = gpu.zeros(&[k_bytes.div_ceil(4)], DType::F32).unwrap();
    let k8_cache = gpu.zeros(&[k8_bytes.div_ceil(4)], DType::F32).unwrap();
    let v_cache = gpu.zeros(&[v_bytes.div_ceil(4)], DType::F32).unwrap();

    let tile = rdna_compute::attention::q8_flash_tile_size(
        &gpu.arch, N_HEADS, N_KV_HEADS, HEAD_DIM, max_seq,
    );
    let partials_len = N_HEADS * max_seq.div_ceil(tile) * (2 + HEAD_DIM);
    eprintln!(
        "  flash tile={tile}, partials tiles={}",
        max_seq.div_ceil(tile)
    );
    eprintln!();

    for seq_len in [32_768usize, 124_345] {
        for batch in [8usize, 16, 32, 64, 128] {
            let q_elems = batch * N_HEADS * HEAD_DIM;
            let q = gpu.upload_f32(&lcg(0xa5, q_elems), &[q_elems]).unwrap();
            let out = gpu.zeros(&[q_elems], DType::F32).unwrap();
            let pos: Vec<i32> = (0..batch).map(|i| (seq_len - batch + i) as i32).collect();
            let positions = gpu.upload_f32(&vec![0.0f32; batch], &[batch]).unwrap();
            let pb: Vec<u8> = pos.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&positions.buf, &pb).unwrap();
            let partials = gpu
                .zeros(
                    &[partials_len.max(batch * N_HEADS * (2 + HEAD_DIM))],
                    DType::F32,
                )
                .unwrap();

            for tier in ["fwht3", "q8"] {
                let run = |gpu: &mut Gpu| {
                    if tier == "fwht3" {
                        gpu.attention_flash_fwht3_batched_masked(
                            &q, &k_cache, &v_cache, &out, &positions, &signs1, &signs2, N_HEADS,
                            N_KV_HEADS, HEAD_DIM, max_seq, seq_len, batch, &partials, None, 0, 0,
                            8,
                        )
                        .expect("fwht3 decode attention");
                    } else {
                        gpu.attention_flash_q8_0_batched_masked(
                            &q, &k8_cache, &v_cache, &out, &positions, N_HEADS, N_KV_HEADS,
                            HEAD_DIM, max_seq, seq_len, batch, &partials, None, 0, 0,
                        )
                        .expect("q8 decode attention");
                    }
                };
                run(&mut gpu);
                gpu.hip.device_synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    run(&mut gpu);
                }
                gpu.hip.device_synchronize().unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

                let per_head = if tier == "fwht3" {
                    k_bytes_per_head()
                } else {
                    v_bytes_per_head()
                };
                let bytes = seq_len * N_KV_HEADS * (per_head + v_bytes_per_head());
                let gbs = bytes as f64 / (ms / 1000.0) / 1e9;
                // Per-token cost is what decode actually pays: one verify step
                // serves `batch` draft slots.
                let us_per_tok = ms * 1000.0 / batch as f64;
                println!(
                    "{seq_len:>9}  {batch:>6}  {tier:>6}  {ms:>9.3}  {gbs:>9.1}  {us_per_tok:>11.1}"
                );
            }
        }
    }

    eprintln!();
    eprintln!("Reading this: R9700 roofline is ~640-700 GB/s (the short-context");
    eprintln!("decode path measures ~660).");
    eprintln!("  * q8 ~1.46x slower than fwht3  => bandwidth-bound by bytes.");
    eprintln!("  * q8 ~= fwht3                  => bytes are NOT the limiter;");
    eprintln!("    fwht3 compression buys nothing inside this kernel.");
    eprintln!("  * us/token falling sharply with batch => occupancy-starved at");
    eprintln!("    the verify batch of 8, the same wall FA2 64..=512 describes.");
}
