//! Issue 015 step 0 — the MLP-zero collapse, reproduced at unit scale (no checkpoint).
//!
//! The 2026-05-20 per-layer diagnostic (`test-nan-per-layer`) showed
//! `Gemma2MLP::forward` producing all-zero output from layer 0 on **both** the
//! f16 and f32 Metal backends while attention produced real signal. These tests
//! isolate the collapse without the 5 GB checkpoint: a random-init MLP at
//! production dims, identical weights copied to an NdArray (CPU) reference,
//! every sub-step of the forward compared.
//!
//! FINDING (2026-09-01): the bug is NOT in any single op and NOT f16-specific.
//! `Metal` is `burn_fusion::Fusion<CubeBackend<WgpuRuntime>>` (burn-wgpu's
//! `fusion` default feature), and the collapse lives in burn-cubecl-fusion's
//! **matmul epilogue fusion**: the exact `Gemma2MLP::forward` op sequence
//! (`cast -> matmul -> cast -> gelu -> cast -> matmul -> mul -> cast -> matmul`)
//! collapses to zeros or silently-wrong values, while:
//!
//! - every individual op matches the CPU reference,
//! - the same sequence WITHOUT the casts matches (V5 probe),
//! - any single cast position matches (V7/V8/V9 probe),
//! - the IDENTICAL sequence on the raw `CubeBackend` (fusion-free) matches.
//!
//! Re-running the identical repro produced DIFFERENT corruption (exact zeros
//! once, wrong-values another run) — consistent with a mis-resolved operand
//! index in the fused epilogue, not deterministic wrong math. The bug is
//! shape-dependent: at small dims a SINGLE fused matmul diverges too, while
//! the raw backend is bit-exact (diff 0.0) on the identical shape+weights.
//!
//! Test layout:
//! - PASSING: sub-step bisect (Metal fusion ops individually == CPU), gelu
//!   extremes, and the raw fusion-free composed forward — these are the
//!   standing regression gates.
//! - `#[ignore]`d: the composed forward on the fusion backend — fails by
//!   construction until the burn-cubecl-fusion epilogue bug is fixed; kept as
//!   the one-command repro (`--ignored`).
//!
//! Run: `cargo test -p lora-gemma2 --features metal --test mlp_zero_bisect`
//! Repro only: `cargo test -p lora-gemma2 --features metal --test mlp_zero_bisect -- --ignored`

use burn::backend::Metal;
use burn::tensor::Tensor;
use burn::tensor::activation::gelu_approximate;
use burn_backend::BackendTypes;
use burn_ndarray::NdArray;

use lora_gemma2::model::{Gemma2MLP, linear_f32};
use lora_gemma2::types::Gemma2Config;

type Nb = NdArray<f32>;

/// Deterministic pseudo-random input in [-8, 8]: plain integer LCG mapped over
/// the full range, identical bits on both backends (from_floats on each device).
fn lcg_input(n: usize) -> Vec<f32> {
    let mut s: u64 = 0x2545F4914F6CDD1D;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 4096) as f32 / 4096.0 - 0.5) * 16.0
        })
        .collect()
}

fn max_abs(t: &Tensor<Metal, 3>) -> f64 {
    t.clone().abs().max().into_scalar() as f64
}

fn max_abs_nd(t: &Tensor<Nb, 3>) -> f64 {
    t.clone().abs().max().into_scalar() as f64
}

/// Max elementwise diff between the Metal and NdArray versions of a sub-step,
/// compared host-side in f64.
fn diff_max(metal: &Tensor<Metal, 3>, nd: &Tensor<Nb, 3>) -> f64 {
    let m = metal
        .clone()
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    let r = nd
        .clone()
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    assert_eq!(m.len(), r.len(), "shape mismatch between backends");
    m.iter()
        .zip(r.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max)
}

/// Copy the Metal MLP's weights into an identical-shape NdArray MLP.
fn copy_weights(metal: &Gemma2MLP<Metal>, nd: &mut Gemma2MLP<Nb>, dev: &<Nb as BackendTypes>::Device) {
    let gate = metal.gate_proj.weight.val().into_data().convert::<f32>();
    let up = metal.up_proj.weight.val().into_data().convert::<f32>();
    let down = metal.down_proj.weight.val().into_data().convert::<f32>();
    nd.gate_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(gate, dev));
    nd.up_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(up, dev));
    nd.down_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(down, dev));
}

/// Assert a Metal sub-step is non-trivial (the collapse signature is max|v| < 5e-5)
/// and matches the CPU reference to collapse-detector tolerance.
fn assert_substep_matches(name: &str, metal: &Tensor<Metal, 3>, nd: &Tensor<Nb, 3>, rel_tol: f64) {
    let m_max = max_abs(metal);
    let r_max = max_abs_nd(nd);
    assert!(
        m_max > 1e-3,
        "{name}: Metal sub-step COLLAPSED (max|v| = {m_max:.3e}, the issue-015 signature)"
    );
    assert!(
        r_max > 1e-3,
        "{name}: NdArray reference collapsed (max|v| = {r_max:.3e}) — test is vacuous"
    );
    let d = diff_max(metal, nd);
    let denom = r_max.max(1e-6);
    assert!(
        d / denom <= rel_tol,
        "{name}: Metal vs NdArray max diff {d:.3e} exceeds {rel_tol:.1e} relative (ref max {r_max:.3e})"
    );
}

/// Sub-step bisect at production dims: every op of `Gemma2MLP::forward` on the
/// fusion Metal backend, driven individually (stream drained between ops),
/// matches the CPU reference. PASSING — the collapse only appears when the ops
/// compose inside one fused stream (see the #[ignore]d test below).
#[test]
fn mlp_substeps_metal_f32_match_ndarray_reference() {
    let dev = <Metal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();
    // Only the MLP dims matter for these tests.
    let config = Gemma2Config::new(256000, 2304, 26, 9216, 8, 4, 256);

    let metal_mlp = Gemma2MLP::<Metal>::new(&config, &dev);
    let mut nd_mlp = Gemma2MLP::<Nb>::new(&config, &nd_dev);
    copy_weights(&metal_mlp, &mut nd_mlp, &nd_dev);

    // Linear weights are randomly initialized — assert they are not zero
    // (a zero-init on this backend would be a distinct root cause).
    let w_max = metal_mlp.gate_proj.weight.val().abs().max().into_scalar();
    assert!(w_max > 1e-3, "Metal Linear init produced zero weights");

    let [b, s, h] = [1usize, 4, 2304];
    let x_vec = lcg_input(b * s * h);
    let x_m = Tensor::<Metal, 1>::from_floats(x_vec.as_slice(), &dev).reshape([b, s, h]);
    let x_n = Tensor::<Nb, 1>::from_floats(x_vec.as_slice(), &nd_dev).reshape([b, s, h]);

    // gate = gelu(linear_f32(gate_proj, x))
    let gate_lin_m = linear_f32(&metal_mlp.gate_proj, x_m.clone());
    let gate_lin_n = linear_f32(&nd_mlp.gate_proj, x_n.clone());
    assert_substep_matches("gate linear (2304->9216)", &gate_lin_m, &gate_lin_n, 1e-2);

    let gelu_m = gelu_approximate(gate_lin_m.clone().cast(burn::tensor::DType::F32));
    let gelu_n = gelu_approximate(gate_lin_n.clone().cast(burn::tensor::DType::F32));
    assert_substep_matches("gelu_approximate", &gelu_m, &gelu_n, 1e-2);

    let up_m = linear_f32(&metal_mlp.up_proj, x_m.clone());
    let up_n = linear_f32(&nd_mlp.up_proj, x_n.clone());
    assert_substep_matches("up linear (2304->9216)", &up_m, &up_n, 1e-2);

    let mul_m = gelu_m.clone() * up_m.clone();
    let mul_n = gelu_n.clone() * up_n.clone();
    assert_substep_matches("gelu(gate) * up", &mul_m, &mul_n, 1e-2);

    // down projection via linear_f32 (9216 -> 2304).
    let down_m = linear_f32(&metal_mlp.down_proj, mul_m.clone());
    let down_n = linear_f32(&nd_mlp.down_proj, mul_n.clone());
    assert_substep_matches("down linear (9216->2304)", &down_m, &down_n, 1e-2);
}

/// THE REPRO: the composed `Gemma2MLP::forward` on the fusion Metal backend
/// collapses while every individual op passes. Fails by construction until the
/// burn-cubecl-fusion matmul-epilogue bug is fixed — run with `--ignored`.
#[test]
#[ignore = "burn-cubecl-fusion fusion bug, INTERMITTENT at this shape (exact-zero collapse observed twice, then passed on a later run — stream/allocator-layout dependent); the deterministic small-shape variant below is the stable repro (riir-burner issue 015, 2026-09-01); passing control: mlp_forward_raw_metal_matches_ndarray"]
fn mlp_forward_fusion_metal_collapse_repro() {
    let dev = <Metal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();
    let config = Gemma2Config::new(256000, 2304, 26, 9216, 8, 4, 256);

    let metal_mlp = Gemma2MLP::<Metal>::new(&config, &dev);
    let mut nd_mlp = Gemma2MLP::<Nb>::new(&config, &nd_dev);
    copy_weights(&metal_mlp, &mut nd_mlp, &nd_dev);

    let [b, s, h] = [1usize, 4, 2304];
    let x_vec = lcg_input(b * s * h);
    let x_m = Tensor::<Metal, 1>::from_floats(x_vec.as_slice(), &dev).reshape([b, s, h]);
    let x_n = Tensor::<Nb, 1>::from_floats(x_vec.as_slice(), &nd_dev).reshape([b, s, h]);

    let fwd_m = metal_mlp.forward(x_m);
    let fwd_n = nd_mlp.forward(x_n);
    assert_substep_matches("Gemma2MLP::forward composed (fusion)", &fwd_m, &fwd_n, 1e-2);
}

/// `gelu_approximate` at extreme magnitudes: tanh saturation must not produce
/// NaN/zero divergence between backends (015 hypothesis: tanh(inner) saturation —
/// REFUTED by this test passing).
#[test]
fn gelu_approximate_extremes_metal_match_ndarray() {
    let dev = <Metal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();

    let vals: Vec<f32> = [
        -1300.0, -120.0, -27.0, -1.0, -0.5, 0.0, 1e-4, 0.5, 1.0, 27.0, 120.0, 1300.0,
    ]
    .to_vec();
    let x_m = Tensor::<Metal, 1>::from_floats(vals.as_slice(), &dev).reshape([1, vals.len(), 1]);
    let x_n = Tensor::<Nb, 1>::from_floats(vals.as_slice(), &nd_dev).reshape([1, vals.len(), 1]);

    let g_m = gelu_approximate(x_m.cast(burn::tensor::DType::F32));
    let g_n = gelu_approximate(x_n.cast(burn::tensor::DType::F32));

    let m: Vec<f64> = g_m
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    let r: Vec<f64> = g_n
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    for (i, (a, b)) in m.iter().zip(r.iter()).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "gelu[{i}] non-finite: {a} vs {b}");
        assert!(
            (a - b).abs() <= 1e-3 + 1e-4 * b.abs(),
            "gelu[{i}] diverged: Metal {a} vs NdArray {b} (input {})",
            vals[i]
        );
    }
}

// ---------------------------------------------------------------------------
// The fusion A/B: identical composed forward on the RAW (fusion-free) backend.
//
// `Metal` = `Fusion<CubeBackend<...>>` (burn-wgpu's `fusion` default feature);
// `RawMetal` strips the fusion wrapper. The raw path PASSES, which localizes
// the bug to burn-fusion / burn-cubecl-fusion stream optimization, not to
// burn-cubecl/wgpu kernels. This is the standing regression gate for the raw
// path AND the control arm of the A/B.
// ---------------------------------------------------------------------------

type RawMetal = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u8>;

fn max_abs_raw(t: &Tensor<RawMetal, 3>) -> f64 {
    t.clone().abs().max().into_scalar() as f64
}

fn diff_max_raw(metal: &Tensor<RawMetal, 3>, nd: &Tensor<Nb, 3>) -> f64 {
    let m = metal
        .clone()
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    let r = nd
        .clone()
        .into_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("f64 vec");
    assert_eq!(m.len(), r.len(), "shape mismatch between backends");
    m.iter()
        .zip(r.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn mlp_forward_raw_metal_matches_ndarray() {
    let dev = <RawMetal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();
    let config = Gemma2Config::new(256000, 2304, 26, 9216, 8, 4, 256);

    let raw_mlp = Gemma2MLP::<RawMetal>::new(&config, &dev);
    let mut nd_mlp = Gemma2MLP::<Nb>::new(&config, &nd_dev);
    let gate = raw_mlp.gate_proj.weight.val().into_data().convert::<f32>();
    let up = raw_mlp.up_proj.weight.val().into_data().convert::<f32>();
    let down = raw_mlp.down_proj.weight.val().into_data().convert::<f32>();
    nd_mlp.gate_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(gate, &nd_dev));
    nd_mlp.up_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(up, &nd_dev));
    nd_mlp.down_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(down, &nd_dev));

    let [b, s, h] = [1usize, 4, 2304];
    let x_vec = lcg_input(b * s * h);
    let x_r = Tensor::<RawMetal, 1>::from_floats(x_vec.as_slice(), &dev).reshape([b, s, h]);
    let x_n = Tensor::<Nb, 1>::from_floats(x_vec.as_slice(), &nd_dev).reshape([b, s, h]);

    let fwd_raw = raw_mlp.forward(x_r);
    let fwd_n = nd_mlp.forward(x_n);

    let raw_max = max_abs_raw(&fwd_raw);
    let nd_max = max_abs_nd(&fwd_n);
    let d = diff_max_raw(&fwd_raw, &fwd_n);
    println!("raw Metal forward: max|v| = {raw_max:.4e}, nd max|v| = {nd_max:.4e}, max diff = {d:.4e}");
    assert!(
        raw_max > 1e-3,
        "RAW Metal (fusion-free) composed forward COLLAPSED (max|v| = {raw_max:.3e}) — burn-cubecl bug, not fusion"
    );
    assert!(
        d / nd_max.max(1e-6) <= 1e-2,
        "RAW Metal diverges from NdArray by {d:.3e} (ref max {nd_max:.3e})"
    );
}

/// Shape-genericity evidence: a SINGLE fused matmul + cast diverges at small
/// dims on the fusion backend (diff 9.3 vs ref max 9.5), while the raw backend
/// is EXACT (diff 0.0) on the same shape/weights — the fusion bug also fires
/// for individual fused matmuls at shapes where vectorization/tiling differs.
/// Fails by construction until the burn-cubecl-fusion bug is fixed; the raw
/// control below is its passing twin.
#[test]
#[ignore = "burn-cubecl-fusion bug, small-shape variant: single linear_f32 diverges at [2,5,64] on the fusion backend while raw CubeBackend is exact (riir-burner issue 015, 2026-09-01)"]
fn mlp_substeps_small_dims_metal_match_ndarray() {
    let dev = <Metal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();
    let config = Gemma2Config::new(1000, 64, 2, 256, 4, 2, 16);

    let metal_mlp = Gemma2MLP::<Metal>::new(&config, &dev);
    let mut nd_mlp = Gemma2MLP::<Nb>::new(&config, &nd_dev);
    copy_weights(&metal_mlp, &mut nd_mlp, &nd_dev);

    let [b, s, h] = [2usize, 5, 64];
    let x_vec = lcg_input(b * s * h);
    let x_m = Tensor::<Metal, 1>::from_floats(x_vec.as_slice(), &dev).reshape([b, s, h]);
    let x_n = Tensor::<Nb, 1>::from_floats(x_vec.as_slice(), &nd_dev).reshape([b, s, h]);

    let gate_m = linear_f32(&metal_mlp.gate_proj, x_m.clone());
    let gate_n = linear_f32(&nd_mlp.gate_proj, x_n.clone());
    assert_substep_matches("small gate linear", &gate_m, &gate_n, 1e-2);

    let up_m = linear_f32(&metal_mlp.up_proj, x_m);
    let up_n = linear_f32(&nd_mlp.up_proj, x_n);
    assert_substep_matches("small up linear", &up_m, &up_n, 1e-2);
}

/// Raw-backend control at the SAME small dims: distinguishes "fusion bug fires
/// at small shapes too" from "burn-cubecl kernel bug at small shapes".
#[test]
fn mlp_substeps_small_dims_raw_metal_match_ndarray() {
    let dev = <RawMetal as BackendTypes>::Device::default();
    let nd_dev = <Nb as BackendTypes>::Device::default();
    let config = Gemma2Config::new(1000, 64, 2, 256, 4, 2, 16);

    let raw_mlp = Gemma2MLP::<RawMetal>::new(&config, &dev);
    let mut nd_mlp = Gemma2MLP::<Nb>::new(&config, &nd_dev);
    let gate = raw_mlp.gate_proj.weight.val().into_data().convert::<f32>();
    let up = raw_mlp.up_proj.weight.val().into_data().convert::<f32>();
    nd_mlp.gate_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(gate, &nd_dev));
    nd_mlp.up_proj.weight = burn::module::Param::from_tensor(Tensor::from_data(up, &nd_dev));

    let [b, s, h] = [2usize, 5, 64];
    let x_vec = lcg_input(b * s * h);
    let x_r = Tensor::<RawMetal, 1>::from_floats(x_vec.as_slice(), &dev).reshape([b, s, h]);
    let x_n = Tensor::<Nb, 1>::from_floats(x_vec.as_slice(), &nd_dev).reshape([b, s, h]);

    let gate_r = linear_f32(&raw_mlp.gate_proj, x_r.clone());
    let gate_n = linear_f32(&nd_mlp.gate_proj, x_n.clone());
    let d_gate = diff_max_raw(&gate_r, &gate_n);
    println!("raw small gate: diff={d_gate:.3e}");

    let up_r = linear_f32(&raw_mlp.up_proj, x_r);
    let up_n = linear_f32(&nd_mlp.up_proj, x_n);
    let d_up = diff_max_raw(&up_r, &up_n);
    let up_max = up_n.clone().abs().max().into_scalar() as f64;
    println!("raw small up: diff={d_up:.3e} ref_max={up_max:.3e}");
    assert!(
        d_up / up_max.max(1e-6) <= 1e-2,
        "RAW Metal small-dims up linear diverges by {d_up:.3e} (ref max {up_max:.3e}) — burn-cubecl kernel bug, not fusion"
    );
}
