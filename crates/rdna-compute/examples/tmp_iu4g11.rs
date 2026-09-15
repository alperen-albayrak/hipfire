// SPDX-License-Identifier: Apache-2.0
// THROWAWAY for exp/iu4-direct-gfx11 (do not upstream as-is).
// Parity: synthetic MQ4V2 weights (valid v2 headers incl. zero-scale
// halves) + CPU-packed int4 X (per-128 MSE-clip grid) vs CPU-f64 oracle on
// the same quantized X. Gate: relL2 <= 1e-5, N in {128,256,512} + M tail.
// Bench: gate_up M=17408 K=5120 N=512 full_set, kernel-only, 3 warmup / 30
// timed, iu4 (occ2 + occ3) vs production X128.
use hip_bridge::KernargBlob;
use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const IU4_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip");
const IU4_MOD: &str = "mmq_iu4_exp";
const IU4_FULL_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set";
const IU4_FULL_SET_OCC3: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_occ3";
const IU4_BASE: &str = "gemm_mq4g256v2_residual_mmq_iu4";
const IU4_QUANT: &str = "quantize_int4_mmq_ds128";
const SHARED_IU4: u32 = (128 * 18 + 128 * 44) * 4;

const GROUP: usize = 256;
const HALF: usize = 128;
const GBYTES: usize = 136;
const CANARY: f32 = 7.654_321;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

struct Weights {
    blob: Vec<u8>,
    /// dequant tables for the oracle: per (row, group, half): (sc, zp)
    sc: Vec<f32>,
    zp: Vec<f32>,
    /// nibble codes per element, row-major m*k
    q: Vec<u8>,
}

/// Synthetic MQ4V2 weights: disjoint half ranges (wrong half-select is
/// unmissable) + every 9th half constant (zero-scale header, codes 0).
fn build_weights(m: usize, k: usize) -> Weights {
    assert_eq!(k % GROUP, 0);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GBYTES];
    let mut sc = vec![0.0f32; m * gpr * 2];
    let mut zp = vec![0.0f32; m * gpr * 2];
    let mut q = vec![0u8; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GBYTES;
            for h in 0..2 {
                let salt = (r * 7919 + g * 104_729 + h * 999_983) as u32;
                let constant = (r + g + h) % 9 == 4;
                let (s_bits, z_bits, s_rt, z_rt) = if constant {
                    let c = 3.25 + (salt % 5) as f32;
                    let zb = half_from_f32(c);
                    (0u16, zb, 0.0f32, f16_to_f32(zb))
                } else {
                    let (lo, hi) = if h == 0 {
                        (-1.0f32, 1.0f32)
                    } else {
                        (96.0f32, 160.0f32)
                    };
                    let step = (hi - lo) / 15.0;
                    let sb = half_from_f32(step);
                    let zb = half_from_f32(lo);
                    (sb, zb, f16_to_f32(sb), f16_to_f32(zb))
                };
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4]
                    .copy_from_slice(&z_bits.to_le_bytes());
                sc[(r * gpr + g) * 2 + h] = s_rt;
                zp[(r * gpr + g) * 2 + h] = z_rt;
                if s_rt != 0.0 {
                    for i in 0..HALF {
                        // deterministic pseudo-weight in [lo, hi] of this half
                        let (lo, hi) = if h == 0 {
                            (-1.0f32, 1.0f32)
                        } else {
                            (96.0f32, 160.0f32)
                        };
                        let w = lo + (hi - lo) * prng(r * k + g * GROUP + h * HALF + i, salt);
                        let code = ((w - z_rt) / s_rt + 0.5).floor().clamp(0.0, 15.0) as u8;
                        q[r * k + g * GROUP + h * HALF + i] = code;
                    }
                }
            }
            // Contiguous nibble packing: even K -> lo, odd K -> hi.
            for i in 0..HALF {
                let a = q[r * k + g * GROUP + 2 * i] & 0xF;
                let b = q[r * k + g * GROUP + 2 * i + 1] & 0xF;
                blob[dst + 8 + i] = a | (b << 4);
            }
        }
    }
    Weights { blob, sc, zp, q }
}

struct PackedX {
    /// 72 B per (block128, col): [f32 d][i32 s][64 B qs]
    bytes: Vec<u8>,
    /// oracle tables
    d: Vec<f32>,
    s: Vec<i32>,
    q: Vec<i8>,
}

/// W4A4 recipe: per-128 amax, 8-candidate MSE-clip grid
/// d_j = (amax/7)*0.5*(1+j/7), strict-< argmin; zero group -> d=1, q=0.
/// Lane-mirrored f32 match of `quantize_int4_mmq_ds128` (f32 grid with
/// explicit single-rounding FMA): wave32 lane l owns group elements
/// 4l..4l+3, per-lane f32 partial MSE, snapshot butterfly (offsets
/// 16..1), per-lane strict-< argmin, d from lane 0. Bit-exactness needs
/// the same op order AND single-rounding FMA on both sides (`mul_add`).
fn pack_int4_x(x: &[f32], n: usize, k: usize) -> PackedX {
    assert_eq!(x.len(), n * k);
    assert_eq!(k % 128, 0);
    let nb = k / 128;
    let mut bytes = vec![0u8; nb * n * 72];
    let mut d = vec![0.0f32; nb * n];
    let mut s = vec![0i32; nb * n];
    let mut q = vec![0i8; n * k];
    for col in 0..n {
        for b in 0..nb {
            let mut amax = 0.0f32;
            for i in 0..128 {
                amax = amax.max(x[col * k + b * 128 + i].abs());
            }
            let (db, ssum) = if amax == 0.0 {
                (1.0f32, 0i32)
            } else {
                // Per-lane quantized picks (lane l owns elements 4l..4l+3).
                let mut lane_d = [1.0f32; 32];
                let mut lane_q = [[0i8; 4]; 32];
                let mut best = [f32::INFINITY; 32];
                for j in 0..8 {
                    let dj = (amax / 7.0) * 0.5 * (1.0 + j as f32 / 7.0);
                    let mut part = [0.0f32; 32];
                    let mut qq = [[0i8; 4]; 32];
                    for l in 0..32 {
                        let mut m = 0.0f32;
                        for e in 0..4 {
                            let v = x[col * k + b * 128 + l * 4 + e];
                            let qf = (v / dj).round_ties_even().clamp(-8.0, 7.0);
                            let qi = qf as i8;
                            qq[l][e] = qi;
                            let err = (-qf).mul_add(dj, v);
                            m = err.mul_add(err, m);
                        }
                        part[l] = m;
                    }
                    // Snapshot butterfly: each lane reads pre-update values.
                    for &off in &[16, 8, 4, 2, 1] {
                        let prev = part;
                        for l in 0..32 {
                            part[l] = prev[l] + prev[l ^ off];
                        }
                    }
                    for l in 0..32 {
                        if part[l] < best[l] {
                            best[l] = part[l];
                            lane_d[l] = dj;
                            lane_q[l] = qq[l];
                        }
                    }
                }
                let mut ss = 0i32;
                for l in 0..32 {
                    for e in 0..4 {
                        q[col * k + b * 128 + l * 4 + e] = lane_q[l][e];
                        ss += lane_q[l][e] as i32;
                    }
                }
                (lane_d[0], ss)
            };
            if amax == 0.0 {
                for i in 0..128 {
                    q[col * k + b * 128 + i] = 0;
                }
            }
            d[b * n + col] = db;
            s[b * n + col] = ssum;
            let off = (b * n + col) * 72;
            bytes[off..off + 4].copy_from_slice(&db.to_le_bytes());
            bytes[off + 4..off + 8].copy_from_slice(&ssum.to_le_bytes());
            for i in 0..64 {
                let a = (q[col * k + b * 128 + 2 * i] as u8) & 0xF;
                let bb = (q[col * k + b * 128 + 2 * i + 1] as u8) & 0xF;
                bytes[off + 8 + i] = a | (bb << 4);
            }
        }
    }
    PackedX { bytes, d, s, q }
}

/// CPU-f64 oracle on the SAME quantized X the kernel reads.
/// Y layout: col-major [N][M], y[n*M + m].
fn oracle(w: &Weights, px: &PackedX, m: usize, k: usize, n: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let nb = k / 128;
    let mut y = vec![0.0f64; n * m];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f64;
            for b in 0..nb {
                let g = b / 2;
                let h = b % 2;
                let sc = w.sc[(row * gpr + g) * 2 + h] as f64;
                let zp = w.zp[(row * gpr + g) * 2 + h] as f64;
                let dd = px.d[b * n + col] as f64;
                let ss = px.s[b * n + col] as f64;
                let mut c: i64 = 0;
                for i in 0..128 {
                    let wq = w.q[row * k + b * 128 + i] as i64;
                    let xq = px.q[col * k + b * 128 + i] as i64;
                    c += wq * xq;
                }
                acc += sc * dd * c as f64 + zp * dd * ss;
            }
            y[col * m + row] = acc;
        }
    }
    y
}

fn rel_l2(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        let e = *a as f64 - *b;
        num += e * e;
        den += *b * *b;
    }
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (num / den).sqrt()
    }
}

fn run_parity_arm(
    gpu: &mut Gpu,
    w: &Weights,
    d_a: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let full = m % 128 == 0 && n % 128 == 0;
    let func = if full { IU4_FULL_SET } else { IU4_BASE };
    // Deterministic X with range variation across columns + one zero column.
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            if col == n - 1 {
                0.0
            } else {
                let scale = 1.0 + (col % 5) as f32;
                (prng(i, 0xC0FF_EE00) * 2.0 - 1.0) * scale
            }
        })
        .collect();
    let px = pack_int4_x(&x, n, k);
    let want = oracle(w, &px, m, k, n);
    let d_xq = gpu.upload_raw(&px.bytes, &[px.bytes.len()]).expect("upload xq");
    let mut y_host = vec![0.0f32; n * m + 1];
    y_host[n * m] = CANARY;
    let d_y = gpu.upload_f32(&y_host, &[n * m + 1]).expect("upload y");
    gpu.ensure_kernel_public(IU4_MOD, IU4_SRC, func).expect("JIT iu4");
    let mut b = KernargBlob::new();
    b.push_ptr(d_a.buf.as_ptr() as *const _);
    b.push_ptr(d_xq.buf.as_ptr() as *const _);
    b.push_ptr(d_y.buf.as_ptr() as *const _);
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    b.push_i32(0);
    let mut blob = b.into_vec();
    let grid = [(m.div_ceil(128)) as u32, (n.div_ceil(128)) as u32, 1];
    gpu.launch_kernel_blob(func, grid, [32, 8, 1], SHARED_IU4, &mut blob)
        .expect("launch iu4");
    gpu.hip.device_synchronize().expect("sync");
    let got_full = gpu.download_f32(&d_y).expect("download");
    let canary_ok =
        got_full.len() == n * m + 1 && got_full[n * m].to_bits() == CANARY.to_bits();
    let got = &got_full[..n * m];
    let r = rel_l2(got, &want);
    let finite = got.iter().all(|v| v.is_finite());
    let ok = canary_ok && finite && r <= 1e-5;
    eprintln!(
        "  [M={m} K={k} N={n} {func}] canary={canary_ok} finite={finite} relL2={r:.3e} [{}]",
        if ok { "PASS" } else { "FAIL" }
    );
    let _ = gpu.free_tensor(d_xq);
    let _ = gpu.free_tensor(d_y);
    ok
}

/// Bit-oracle for the GPU int4 pre-pass: CPU-pack X with `pack_int4_x`,
/// run `quantize_int4_mmq_ds128` on-device, and require byte-identical
/// output (72 B per block: d, s, qs).
fn run_quant_oracle_arm(gpu: &mut Gpu, n: usize, k: usize) -> bool {
    assert_eq!(k % 128, 0);
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            if col == n - 1 {
                0.0
            } else {
                let scale = 1.0 + (col % 7) as f32 * 0.5;
                (prng(i, 0x51A7_0E00) * 2.0 - 1.0) * scale
            }
        })
        .collect();
    let px = pack_int4_x(&x, n, k);
    let d_x = gpu.upload_f32(&x, &[n * k]).expect("upload x");
    let nb = k / 128;
    // F32 tensor (not raw): download_f32 sizes by numel, so allocate
    // nb*n*72/4 f32 words covering the same bytes.
    let d_y = gpu
        .upload_f32(&vec![0.0f32; nb * n * 72 / 4], &[nb * n * 72 / 4])
        .expect("alloc y");
    gpu.ensure_kernel_public(IU4_MOD, IU4_SRC, IU4_QUANT)
        .expect("JIT quant");
    let mut b = KernargBlob::new();
    b.push_ptr(d_x.buf.as_ptr() as *const _);
    b.push_ptr(d_y.buf.as_ptr() as *const _);
    b.push_i32(k as i32);
    b.push_i32(n as i32);
    let mut blob = b.into_vec();
    let grid = [((k + 1023) / 1024) as u32, n as u32, 1];
    gpu.launch_kernel_blob(IU4_QUANT, grid, [256, 1, 1], 0, &mut blob)
        .expect("launch quant");
    gpu.hip.device_synchronize().expect("sync");
    let got_f32 = gpu.download_f32(&d_y).expect("download");
    let got: &[u8] = unsafe {
        std::slice::from_raw_parts(
            got_f32.as_ptr() as *const u8,
            got_f32.len() * std::mem::size_of::<f32>(),
        )
    };
    let mut mism = 0usize;
    let mut first = None;
    for (i, (a, b)) in got.iter().zip(px.bytes.iter()).enumerate() {
        if a != b {
            if first.is_none() {
                first = Some((i, *a, *b));
            }
            mism += 1;
        }
    }
    let ok = mism == 0 && got.len() == px.bytes.len();
    eprintln!(
        "  [quant-oracle N={n} K={k}] bytes={} mism={mism} first={first:?} [{}]",
        got.len(),
        if ok { "PASS" } else { "FAIL" }
    );
    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_y);
    ok
}

fn bench_kernel(
    gpu: &mut Gpu,
    label: &str,
    func: &str,
    src: &str,
    module: &str,
    d_a: *const std::ffi::c_void,
    d_x: *const std::ffi::c_void,
    d_y: *const std::ffi::c_void,
    m: usize,
    k: usize,
    n: usize,
    shared: u32,
    extra_i32: i32,
) -> Vec<f64> {
    gpu.ensure_kernel_public(module, src, func).expect("JIT bench kernel");
    let mk_blob = || {
        let mut b = KernargBlob::new();
        b.push_ptr(d_a);
        b.push_ptr(d_x);
        b.push_ptr(d_y);
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(n as i32);
        b.push_i32(extra_i32);
        b.into_vec()
    };
    let grid = [(m.div_ceil(128)) as u32, (n.div_ceil(128)) as u32, 1];
    let mut blob = mk_blob();
    for _ in 0..3 {
        gpu.launch_kernel_blob(func, grid, [32, 8, 1], shared, &mut blob)
            .expect("warmup");
    }
    gpu.hip.device_synchronize().expect("sync warmup");
    let mut secs = Vec::with_capacity(30);
    for _ in 0..30 {
        let mut b2 = mk_blob();
        let t0 = Instant::now();
        gpu.launch_kernel_blob(func, grid, [32, 8, 1], shared, &mut b2)
            .expect("timed");
        gpu.hip.device_synchronize().expect("sync timed");
        secs.push(t0.elapsed().as_secs_f64());
    }
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops: Vec<f64> = secs.iter().map(|s| flops / s / 1e12).collect();
    eprintln!("  {label}: median run ready (30 calls)");
    tflops
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_bench = args.iter().any(|a| a == "bench");
    let only = args.iter().find_map(|a| {
        if a == "iu4" || a == "x128" || a == "occ3" {
            Some(a.clone())
        } else {
            None
        }
    });
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let do_quant = args.iter().any(|a| a == "quant-oracle");
    if do_quant {
        if gpu.arch != "gfx1100" && gpu.arch != "gfx1151" {
            eprintln!("SKIP: arch {} is not gfx1100/gfx1151", gpu.arch);
            return;
        }
        eprintln!("iu4 pre-pass quant-oracle on {}", gpu.arch);
        let mut all_ok = true;
        for &(n, k) in &[(128usize, 512usize), (100, 512), (256, 1024), (512, 5120)] {
            if !run_quant_oracle_arm(&mut gpu, n, k) {
                all_ok = false;
            }
        }
        if all_ok {
            eprintln!("PASS: quant-oracle bit-exact on all arms");
        } else {
            eprintln!("FAIL: quant-oracle mismatch");
            std::process::exit(1);
        }
        return;
    }
    if gpu.arch != "gfx1100" {
        eprintln!("SKIP: arch {} is not gfx1100", gpu.arch);
        return;
    }
    if !do_bench {
        eprintln!("iu4-direct parity on gfx1100");
        let k = 512;
        let mut all_ok = true;
        for &m in &[128usize, 96usize] {
            let w = build_weights(m, k);
            eprintln!(
                "packed MQ4V2 {} B (M={m} K={k}, disjoint halves + zero-scale)",
                w.blob.len()
            );
            let d_a = gpu.upload_raw(&w.blob, &[w.blob.len()]).expect("upload A");
            for &n in &[128usize, 256usize, 512usize] {
                if !run_parity_arm(&mut gpu, &w, &d_a, m, k, n) {
                    all_ok = false;
                }
            }
            let _ = gpu.free_tensor(d_a);
        }
        if all_ok {
            eprintln!("PASS: all parity arms relL2<=1e-5");
        } else {
            eprintln!("FAIL: parity gate violated");
            std::process::exit(1);
        }
        return;
    }
    // ---- bench: gate_up M=17408 K=5120 N=512 full_set, kernel-only ----
    let (m, k, n) = (17408usize, 5120usize, 512usize);
    eprintln!("iu4-direct bench M={m} K={k} N={n} on {}", gpu.arch);
    let w = build_weights(m, k);
    let d_a = gpu.upload_raw(&w.blob, &[w.blob.len()]).expect("upload A");
    // iu4 X (CPU int4 pack)
    let x: Vec<f32> = (0..n * k)
        .map(|i| (prng(i, 0xC0FF_EE00) * 2.0 - 1.0) * (1.0 + (i % 7) as f32 * 0.25))
        .collect();
    let px = pack_int4_x(&x, n, k);
    let d_xq = gpu.upload_raw(&px.bytes, &[px.bytes.len()]).expect("upload xq");
    // x128 X via production prelude
    let d_xf = gpu.upload_f32(&x, &[n * k]).expect("upload xf");
    let xq_ptr = gpu.ensure_q8_1_mmq_x128(&d_xf, n, k).expect("x128 prelude");
    // ensure_q8_1_mmq_x128 returns a raw device pointer; pass through directly.
    let d_xq8_ptr = xq_ptr as *const std::ffi::c_void;
    let d_y = gpu.zeros(&[n * m], DType::F32).expect("alloc y");
    gpu.hip.device_synchronize().expect("sync setup");
    const BASE_SRC: &str =
        include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq.hip");
    const BASE_MOD: &str = "gemm_mq4g256v2_residual_mmq";
    const BASE_FN: &str = "gemm_mq4g256v2_residual_mmq_full_set_x128";
    const SHARED_BASE: u32 = (128 * 36 + 128 * 76) * 4;
    let run_iu4 = only.is_none() || only.as_deref() == Some("iu4");
    let run_occ3 = only.is_none() || only.as_deref() == Some("occ3");
    let run_x = only.is_none() || only.as_deref() == Some("x128");
    // ABBA within process: caller runs 3 fresh processes; order alternates
    // by round. Round order from argv: "order=iu4,x128" or default x128,iu4.
    let order_iu4_first = args.iter().any(|a| a == "order=iu4,x128");
    for round in 0..2 {
        let iu4_first = if round == 0 { order_iu4_first } else { !order_iu4_first };
        eprintln!("round {round} iu4_first={iu4_first}");
        let mut go = |iu4: bool| {
            if iu4 {
                if run_iu4 {
                    let t = bench_kernel(
                        &mut gpu, "iu4-occ2", IU4_FULL_SET, IU4_SRC, IU4_MOD,
                        d_a.buf.as_ptr() as *const _,
                        d_xq.buf.as_ptr() as *const _,
                        d_y.buf.as_ptr() as *const _, m, k, n, SHARED_IU4, 0,
                    );
                    eprintln!("IU4OCC2 med={:.2} min={:.2} max={:.2}", median(t.clone()), t.iter().cloned().fold(f64::INFINITY, f64::min), t.iter().cloned().fold(0.0f64, f64::max));
                    for v in &t {
                        println!("tflops iu4-occ2 {v:.4}");
                    }
                }
                if run_occ3 {
                    let t = bench_kernel(
                        &mut gpu, "iu4-occ3", IU4_FULL_SET_OCC3, IU4_SRC, IU4_MOD,
                        d_a.buf.as_ptr() as *const _,
                        d_xq.buf.as_ptr() as *const _,
                        d_y.buf.as_ptr() as *const _, m, k, n, SHARED_IU4, 0,
                    );
                    eprintln!("IU4OCC3 med={:.2} min={:.2} max={:.2}", median(t.clone()), t.iter().cloned().fold(f64::INFINITY, f64::min), t.iter().cloned().fold(0.0f64, f64::max));
                    for v in &t {
                        println!("tflops iu4-occ3 {v:.4}");
                    }
                }
            } else if run_x {
                let t = bench_kernel(
                    &mut gpu, "x128", BASE_FN, BASE_SRC, BASE_MOD,
                    d_a.buf.as_ptr() as *const _,
                    d_xq8_ptr,
                    d_y.buf.as_ptr() as *const _, m, k, n, SHARED_BASE, 0,
                );
                eprintln!("X128 med={:.2} min={:.2} max={:.2}", median(t.clone()), t.iter().cloned().fold(f64::INFINITY, f64::min), t.iter().cloned().fold(0.0f64, f64::max));
                for v in &t {
                    println!("tflops x128 {v:.4}");
                }
            }
        };
        if iu4_first {
            go(true);
            go(false);
        } else {
            go(false);
            go(true);
        }
    }
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_xq);
    let _ = gpu.free_tensor(d_xf);
    let _ = gpu.free_tensor(d_y);
    eprintln!("bench done");
}
