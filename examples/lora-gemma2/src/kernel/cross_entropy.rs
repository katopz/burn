//! Fused cross-entropy loss kernel for Gemma 2 training speed optimization.
//!
//! Fuses multiple operations into single GPU kernel dispatches:
//! - **Forward**: numerically stable logsumexp + CE loss in one kernel
//! - **Backward**: softmax + gradient computation in one kernel
//!
//! Replaces the standard burn tensor op chain:
//! `log_softmax → gather → neg → mask_fill → sum → div`
//! (15+ kernel launches) with just 2 (forward + backward).
//!
//! The key win: avoids materializing the full log_probs tensor
//! `[batch*seq, 256K vocab]` in f32 (~4GB for seq=2048).
//!
//! Each cube (workgroup) handles one row (one token position).
//! Threads cooperate via shared memory for the logsumexp reduction.
//!
//! # Reference
//! - `unsloth/kernels/cross_entropy_loss.py` — Triton kernel adapted from
//! - `burn/crates/burn-cubecl/src/kernel/ctc.rs` — shared memory + `sync_cube` patterns

use burn_backend::{Shape, TensorMetadata};
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default thread count per cube (power-of-2 for tree reduction).
const CE_BLOCK_SIZE: usize = 256;

/// Large negative sentinel for initial max.
/// Cannot use true -inf: WGSL rejects `f32(-inf)` as identifier,
/// and f16 max magnitude is ~65504. Matches CTC kernel convention.
const NEG_INF_SENTINEL: f32 = -6.0e4;

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------

/// Fused cross-entropy forward: logsumexp + CE loss.
///
/// Each cube handles one row of `[rows, vocab]`. Threads cooperate via shared
/// memory to compute:
/// 1. Row-wise max (numerical stability)
/// 2. `logsumexp = max + ln(sum(exp(x - max)))`
/// 3. `CE loss = logsumexp - x[label]`  (0.0 for padding / label == -100)
///
/// Logit softcapping is NOT fused here — apply it before calling.
/// See module docs for why.
#[cube(launch)]
fn fused_ce_forward_kernel<F: Float, I: Numeric>(
    logits: &Tensor<F>,
    targets: &Tensor<I>,
    loss: &mut Tensor<F>,
    logsumexp_out: &mut Tensor<F>,
    #[comptime] block_size: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let n_rows = logits.shape(0);
    let vocab_size = logits.shape(1);

    if row >= n_rows {
        terminate!();
    }

    let row_stride = logits.stride(0);
    let col_stride = logits.stride(1);
    let base = row * row_stride;

    let mut shared = SharedMemory::<F>::new(block_size);

    // ---- Phase 1: Row-wise max (numerical stability for logsumexp) ----
    let mut local_max = F::new(NEG_INF_SENTINEL);

    let mut col = tid;
    while col < vocab_size {
        let val = logits[base + col * col_stride];
        local_max = F::max(local_max, val);
        col += block_size;
    }

    shared[tid] = local_max;
    sync_cube();

    // Tree reduction for max
    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            shared[tid] = F::max(shared[tid], shared[tid + stride]);
        }
        stride /= 2;
    }
    sync_cube();

    let row_max = shared[0];

    // ---- Phase 2: sum(exp(x - max)) for logsumexp ----
    let mut local_sum = F::new(0.0);

    col = tid;
    while col < vocab_size {
        let val = logits[base + col * col_stride];
        local_sum = local_sum + (val - row_max).exp();
        col += block_size;
    }

    shared[tid] = local_sum;
    sync_cube();

    // Tree reduction for sum
    stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            shared[tid] = shared[tid] + shared[tid + stride];
        }
        stride /= 2;
    }
    sync_cube();

    // ---- Phase 3: Compute loss and logsumexp (thread 0 only) ----
    if tid == 0 {
        let lse = row_max + shared[0].ln();
        logsumexp_out[row] = lse;

        let label_idx = targets[row];
        let label_i32 = i32::cast_from(label_idx);

        if label_i32 == -100 {
            // Padding token — zero loss, still record logsumexp for backward safety
            loss[row] = F::new(0.0);
        } else {
            let label_pos = u32::cast_from(label_idx) as usize;
            let x = logits[base + label_pos * col_stride];
            loss[row] = lse - x;
        }
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

/// Fused cross-entropy backward: softmax + gradient computation.
///
/// Each cube handles one row. Each thread independently computes gradients
/// for its slice of the vocab dimension — no shared memory or sync needed.
///
/// Gradient formula:
/// ```text
/// grad_logits[i] = dloss * (softmax(logits)[i] - delta(i, label))
/// ```
///
/// For padding tokens (label == -100), all gradients are zero.
#[cube(launch)]
fn fused_ce_backward_kernel<F: Float, I: Numeric>(
    logits: &Tensor<F>,
    grad_loss: &Tensor<F>,
    logsumexp: &Tensor<F>,
    targets: &Tensor<I>,
    grad_logits: &mut Tensor<F>,
    #[comptime] block_size: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let row = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let n_rows = logits.shape(0);
    let vocab_size = logits.shape(1);

    if row >= n_rows {
        terminate!();
    }

    let row_stride = logits.stride(0);
    let col_stride = logits.stride(1);
    let base = row * row_stride;

    let label_idx = targets[row];
    let label_i32 = i32::cast_from(label_idx);
    let is_ignore = label_i32 == -100;
    let dloss = if is_ignore {
        F::new(0.0)
    } else {
        grad_loss[row]
    };
    let lse = logsumexp[row];
    let label_pos = u32::cast_from(label_idx) as usize;

    // Each thread strides over vocab elements — no sync needed
    let mut col = tid;
    while col < vocab_size {
        let x = logits[base + col * col_stride];

        // Softmax: exp(x - logsumexp)
        let mut grad = (x - lse).exp();

        // Subtract 1 at label position: d/dx_i CE = softmax_i - delta(i, label)
        if col == label_pos && !is_ignore {
            grad = grad - F::new(1.0);
        }

        grad_logits[base + col * col_stride] = dloss * grad;
        col += block_size;
    }
}

// ---------------------------------------------------------------------------
// Host-side launch functions
// ---------------------------------------------------------------------------

/// Compute fused cross-entropy loss (forward).
///
/// Replaces: `log_softmax → gather → neg` with a single kernel dispatch.
///
/// # Arguments
/// * `logits`  — Input logits `[rows, vocab_size]` (must be contiguous, f32 recommended)
/// * `targets` — Target class indices `[rows]` (integer tensor; -100 = ignore/padding)
///
/// # Returns
/// `(loss [rows], logsumexp [rows])` — `logsumexp` is saved for the backward pass.
///
/// # Panics
/// Panics if `logits` is not 2-D or `targets` is not 1-D.
pub fn fused_ce_forward<R: CubeRuntime>(
    logits: CubeTensor<R>,
    targets: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let logits = into_contiguous(logits);
    let targets = into_contiguous(targets);

    let [n_rows, _vocab_size] = logits.shape().dims::<2>();
    let client = logits.client.clone();
    let device = logits.device.clone();
    let f_dtype = logits.dtype;
    let i_dtype = targets.dtype;

    let loss = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows]),
        f_dtype,
    );
    let logsumexp = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows]),
        f_dtype,
    );

    // Use hardware max or our default, whichever is smaller (power-of-2 for tree reduction)
    let hw_max = client.properties().hardware.max_cube_dim.0 as usize;
    let block_size = CE_BLOCK_SIZE.min(hw_max).next_power_of_two();

    let cube_count = CubeCount::Static(n_rows as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    fused_ce_forward_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        logits.into_tensor_arg(),
        targets.into_tensor_arg(),
        loss.clone().into_tensor_arg(),
        logsumexp.clone().into_tensor_arg(),
        block_size,
        [f_dtype.into(), i_dtype.into()],
    );

    (loss, logsumexp)
}

/// Compute fused cross-entropy backward pass.
///
/// Produces `d(loss)/d(logits)` in a single kernel dispatch.
///
/// # Arguments
/// * `logits`    — Original logits `[rows, vocab_size]`
/// * `grad_loss` — Upstream gradient `[rows]`
/// * `logsumexp` — Saved from forward `[rows]`
/// * `targets`   — Target class indices `[rows]`
///
/// # Returns
/// Gradient w.r.t. logits `[rows, vocab_size]`
pub fn fused_ce_backward<R: CubeRuntime>(
    logits: CubeTensor<R>,
    grad_loss: CubeTensor<R>,
    logsumexp: CubeTensor<R>,
    targets: CubeTensor<R>,
) -> CubeTensor<R> {
    let logits = into_contiguous(logits);
    let grad_loss = into_contiguous(grad_loss);
    let logsumexp = into_contiguous(logsumexp);
    let targets = into_contiguous(targets);

    let [n_rows, vocab_size] = logits.shape().dims::<2>();
    let client = logits.client.clone();
    let device = logits.device.clone();
    let f_dtype = logits.dtype;
    let i_dtype = targets.dtype;

    let grad_logits = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows, vocab_size]),
        f_dtype,
    );

    let hw_max = client.properties().hardware.max_cube_dim.0 as usize;
    let block_size = CE_BLOCK_SIZE.min(hw_max).next_power_of_two();

    let cube_count = CubeCount::Static(n_rows as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    fused_ce_backward_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        logits.into_tensor_arg(),
        grad_loss.into_tensor_arg(),
        logsumexp.into_tensor_arg(),
        targets.into_tensor_arg(),
        grad_logits.clone().into_tensor_arg(),
        block_size,
        [f_dtype.into(), i_dtype.into()],
    );

    grad_logits
}
