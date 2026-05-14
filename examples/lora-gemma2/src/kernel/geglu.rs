//! Fused GeGLU backward kernel for MLP gradient computation.
//!
//! Computes the backward pass of GeGLU (Gaussian Error Gated Linear Unit)
//! in a single GPU dispatch, avoiding materialization of ~15 intermediate
//! `[N, d_mid]` tensors.
//!
//! # GeGLU forward
//!
//! `h = gelu(e) * g` where `gelu` is the approximate (tanh) GELU.
//!
//! # Backward computation
//!
//! Given upstream gradient `dh`, recomputes:
//! 1. `h  = gelu(e) * g`          — forward output (needed for down LoRA grad)
//! 2. `dg = dh * gelu(e)`         — gradient for up path
//! 3. `de = dh * gelu'(e) * g`    — gradient for gate path
//!
//! # Approximate GELU
//!
//! ```text
//! a = sqrt(2/pi) * (e + 0.044715 * e³)
//! tanh_a = tanh(a)
//! gelu(e) = 0.5 * e * (1 + tanh_a)
//! gelu'(e) = 0.5*(1+tanh_a) + 0.5*e*(1-tanh_a²)*sqrt(2/pi)*(1+0.134145*e²)
//! ```
//!
//! Each thread processes one element independently (no shared memory, no sync).
//!
//! # Reference
//! - `unsloth` fused MLP kernels — Triton kernels we adapted from

use burn_backend::{Shape, TensorMetadata};
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim};

// ---------------------------------------------------------------------------
// Constants (f32 for cubecl F::new compatibility)
// ---------------------------------------------------------------------------

/// sqrt(2/pi) for approximate GELU.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// 0.044715 coefficient for approximate GELU.
const GELU_COEFF: f32 = 0.044715;

/// 3 * 0.044715 = 0.134145 for approximate GELU derivative.
const GELU_COEFF_3: f32 = 0.134145;

/// Default thread count per cube for elementwise operations.
const BLOCK_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

/// Elementwise GeGLU backward kernel.
///
/// Each thread processes one element independently (no shared memory, no sync).
/// Computes (h, dg, de) from (dh, e, g) using approximate GELU and its derivative.
#[cube(launch)]
fn geglu_backward_kernel<F: Float>(
    dh: &Tensor<F>,
    e: &Tensor<F>,
    g: &Tensor<F>,
    h: &mut Tensor<F>,
    dg: &mut Tensor<F>,
    de: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS_X as usize;
    let n_elements = dh.shape(0) * dh.shape(1);

    if idx >= n_elements {
        terminate!();
    }

    // Constants
    let sqrt_2_over_pi = F::new(SQRT_2_OVER_PI);
    let gelu_coeff = F::new(GELU_COEFF);
    let gelu_coeff_3 = F::new(GELU_COEFF_3);
    let half = F::new(0.5);
    let one = F::new(1.0);

    // Load inputs
    let dh_val = dh[idx];
    let e_val = e[idx];
    let g_val = g[idx];

    // ---- Approximate GELU ----
    // a = sqrt(2/pi) * (e + 0.044715 * e^3)
    let e_cubed = e_val * e_val * e_val;
    let a = sqrt_2_over_pi * (e_val + gelu_coeff * e_cubed);
    let tanh_a = a.tanh();
    // gelu(e) = 0.5 * e * (1 + tanh(a))
    let gelu_e = half * e_val * (one + tanh_a);

    // ---- GELU derivative ----
    // gelu'(e) = 0.5*(1+tanh_a) + 0.5*e*(1-tanh_a^2)*sqrt(2/pi)*(1+0.134145*e^2)
    let tanh_a_sq = tanh_a * tanh_a;
    let e_sq = e_val * e_val;
    let gelu_prime = half * (one + tanh_a)
        + half * e_val * (one - tanh_a_sq) * sqrt_2_over_pi * (one + gelu_coeff_3 * e_sq);

    // ---- Write outputs ----
    // Recomputed forward output (needed for down LoRA gradient)
    h[idx] = gelu_e * g_val;
    // Gradient for up projection path
    dg[idx] = dh_val * gelu_e;
    // Gradient for gate projection path
    de[idx] = dh_val * gelu_prime * g_val;
}

// ---------------------------------------------------------------------------
// Host-side launch function
// ---------------------------------------------------------------------------

/// Compute fused GeGLU backward: (h, dg, de) from (dh, e, g) in a single dispatch.
///
/// # Arguments
/// * `dh` — `[N, d_mid]` upstream gradient
/// * `e`  — `[N, d_mid]` gate pre-activation
/// * `g`  — `[N, d_mid]` up projection output
///
/// # Returns
/// `(h [N, d_mid], dg [N, d_mid], de [N, d_mid])` — recomputed forward and gradients.
pub fn geglu_backward<R: CubeRuntime>(
    dh: CubeTensor<R>,
    e: CubeTensor<R>,
    g: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let dh = into_contiguous(dh);
    let e = into_contiguous(e);
    let g = into_contiguous(g);

    let [n_rows, d_mid] = dh.shape().dims::<2>();
    let total = n_rows * d_mid;

    let client = dh.client.clone();
    let device = dh.device.clone();
    let dtype = dh.dtype;

    let h = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows, d_mid]),
        dtype,
    );
    let dg = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows, d_mid]),
        dtype,
    );
    let de = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows, d_mid]),
        dtype,
    );

    let cube_dim = CubeDim::new_1d(BLOCK_SIZE as u32);
    let num_cubes = total.div_ceil(BLOCK_SIZE);
    let cube_count = CubeCount::Static(num_cubes as u32, 1, 1);

    geglu_backward_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        dh.into_tensor_arg(),
        e.into_tensor_arg(),
        g.into_tensor_arg(),
        h.clone().into_tensor_arg(),
        dg.clone().into_tensor_arg(),
        de.clone().into_tensor_arg(),
        dtype.into(),
    );

    (h, dg, de)
}
