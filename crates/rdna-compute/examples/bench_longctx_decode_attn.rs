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
        "{:>9}  {:>10}  {:>9}  {:>9}  {:>8}",
        "seq_len", "max_seq", "ms/call", "GB/s", "arm"
    );

    let q_elems = BATCH * N_HEADS * HEAD_DIM;
    let out_elems = BATCH * N_HEADS * HEAD_DIM;
    // FWHT-256 sign tables, as the fwht3 K-write used.
    let signs1 = gpu.upload_f32(&vec![1.0f32; 256], &[256]).unwrap();
    let signs2 = gpu.upload_f32(&vec![1.0f32; 256], &[256]).unwrap();

    for seq_len in [4096usize, 16_384, 32_768, 65_536, 124_345] {
        // Two arms at the same seq_len: production allocation vs one sized to
        // the request. Only `max_seq` differs, so any gap is candidate B.
        for (arm, max_seq) in [("prod", PROD_MAX_SEQ), ("tight", seq_len.next_multiple_of(256))] {
            if max_seq < seq_len {
                continue;
            }
            let k_bytes = max_seq * N_KV_HEADS * k_bytes_per_head();
            let v_bytes = max_seq * N_KV_HEADS * v_bytes_per_head();
            let k_cache = gpu.zeros(&[k_bytes.div_ceil(4)], DType::F32).unwrap();
            let v_cache = gpu.zeros(&[v_bytes.div_ceil(4)], DType::F32).unwrap();
            let q = gpu.upload_f32(&lcg(0xa5, q_elems), &[q_elems]).unwrap();
            let out = gpu.zeros(&[out_elems], DType::F32).unwrap();

            // Every verify row sits at the end of the context, as a real
            // decode step does.
            let pos: Vec<i32> = (0..BATCH).map(|i| (seq_len - BATCH + i) as i32).collect();
            let positions = gpu.upload_f32(&vec![0.0f32; BATCH], &[BATCH]).unwrap();
            let pb: Vec<u8> = pos.iter().flat_map(|v| v.to_ne_bytes()).collect();
            gpu.hip.memcpy_htod(&positions.buf, &pb).unwrap();

            let tile = rdna_compute::attention::q8_flash_tile_size(
                &gpu.arch, N_HEADS, N_KV_HEADS, HEAD_DIM, max_seq,
            );
            let partials_len = N_HEADS * max_seq.div_ceil(tile) * (2 + HEAD_DIM);
            let partials = gpu.zeros(&[partials_len], DType::F32).unwrap();

            let call = |gpu: &mut Gpu| {
                gpu.attention_flash_fwht3_batched_masked(
                    &q,
                    &k_cache,
                    &v_cache,
                    &out,
                    &positions,
                    &signs1,
                    &signs2,
                    N_HEADS,
                    N_KV_HEADS,
                    HEAD_DIM,
                    max_seq,
                    seq_len,
                    BATCH,
                    &partials,
                    None,
                    0,
                    0,
                    8, // v_mode_bits: Q8_0 V
                )
                .expect("decode attention");
            };

            // Warm the JIT and any lazy allocation before timing.
            call(&mut gpu);
            gpu.hip.device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                call(&mut gpu);
            }
            gpu.hip.device_synchronize().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // One layer reads the whole KV span once.
            let bytes = seq_len * N_KV_HEADS * (k_bytes_per_head() + v_bytes_per_head());
            let gbs = bytes as f64 / (ms / 1000.0) / 1e9;
            println!("{seq_len:>9}  {max_seq:>10}  {ms:>9.3}  {gbs:>9.1}  {arm:>8}");
        }
    }

    eprintln!();
    eprintln!("Reading this: R9700 roofline is ~640-700 GB/s (the short-context");
    eprintln!("decode path measures ~660). A flat low GB/s across seq_len points");
    eprintln!("at candidate A (alignment); a prod-vs-tight gap points at B");
    eprintln!("(partials sized by max_seq). Both can be true.");
}
