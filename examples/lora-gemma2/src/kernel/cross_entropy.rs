//! Fused cross-entropy loss kernel with optional logit softcapping for Gemma 2.
//!
//! Fuses multiple operations into single GPU kernel dispatches:
//! - **Forward**: logsumexp + CE loss, optionally with `softcap * tanh(x / softcap)`
//! - **Backward**: softmax + gradient, with softcapping chain rule
//!
//! Replaces the standard burn tensor op chain:
//! `cast_f32 → div_scalar → tanh → mul_scalar → cast → log_softmax → gather → neg`
//! (15+ kernel launches) with just 2 (forward + backward).
//!
//! When `softcap` is `Some(v)`:
//! - Forward: applies `v * tanh(x / v)` inline during max/logsumexp computation
//! - Backward: applies chain rule `1 - tanh²(x / v)` to the gradient
//!
//! This avoids materializing the softcapped logits tensor `[batch*seq, 256K vocab]`,
//! saving ~4GB for seq=2048 with f32.
//!
//! Each cube (workgroup) handles one row (one token position).
//! Threads cooperate via shared memory for the logsumexp reduction.
//!
//! # Kernel design notes
//!
//! Uses `for` range loops exclusively (no `while` loops with mutable
//! loop variables). Cubecl treats local variables as SSA after first
//! assignment in `while` bodies, causing "mutable operation on const
//! variable" panics. `for` loops avoid this because cubecl manages
//! the loop variable internally.
//!
//! Softcapping uses comptime `do_softcap: bool` for dead code elimination.
//! When disabled, all softcapping branches are removed by the compiler.
//!
//! # Reference
//! - `unsloth/kernels/cross_entropy_loss.py` — Triton kernel adapted from
//! - `burn/crates/burn-cubecl/src/kernel/ctc.rs` — shared memory patterns

use burn_backend::{DType, Shape, TensorMetadata};
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::{cast, into_contiguous};
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
/// f16 max magnitude is ~65504. Matches CTC kernel convention.
const NEG_INF_SENTINEL: f32 = -6.0e4;

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------

/// Fused cross-entropy forward: logsumexp + CE loss with optional softcapping.
///
/// Each cube handles one row of `[rows, vocab]`. Threads cooperate via shared
/// memory to compute:
/// 1. Row-wise max of (optionally softcapped) logits (numerical stability)
/// 2. `logsumexp = max + ln(sum(exp(sc(x) - max)))`
/// 3. `CE loss = logsumexp - sc(x)[label]`  (0.0 for padding / label == -100)
///
/// When `do_softcap` is false, behaves identically to standard cross-entropy.
/// The comptime bool ensures dead code elimination removes softcapping overhead.
#[cube(launch)]
fn fused_ce_forward_kernel<F: Float, I: Numeric>(
    logits: &Tensor<F>,
    targets: &Tensor<I>,
    loss: &mut Tensor<F>,
    logsumexp_out: &mut Tensor<F>,
    #[comptime] block_size: usize,
    #[comptime] do_softcap: bool,
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

    // Gemma 2 2B final logit softcapping: 30.0 * tanh(x / 30.0)
    // Hardcoded as comptime constant — dead code eliminated when do_softcap=false.
    let cap = F::new(30.0);

    let mut shared = SharedMemory::<F>::new(block_size);

    // Number of strided iterations to cover the full vocab dimension.
    // +1 ensures we cover partial blocks at the end (bounds-check inside loop).
    let n_steps = vocab_size / block_size + 1;

    // ---- Phase 1: Row-wise max (of softcapped logits) ----
    shared[tid] = F::new(NEG_INF_SENTINEL);

    for step in 0..n_steps {
        let idx = tid + step * block_size;
        if idx < vocab_size {
            let raw = logits[base + idx * col_stride];
            let val = if do_softcap {
                cap * (raw / cap).tanh()
            } else {
                raw
            };
            let current = shared[tid];
            shared[tid] = F::max(current, val);
        }
    }

    sync_cube();

    // Tree reduction for max — fixed 8 stages handles block_size up to 256.
    tree_reduce_max(shared, tid, block_size);

    sync_cube();

    let row_max = shared[0];

    // ---- Phase 2: sum(exp(sc(x) - max)) for logsumexp ----
    shared[tid] = F::new(0.0);

    for step in 0..n_steps {
        let idx = tid + step * block_size;
        if idx < vocab_size {
            let raw = logits[base + idx * col_stride];
            let val = if do_softcap {
                cap * (raw / cap).tanh()
            } else {
                raw
            };
            let exp_val = (val - row_max).exp();
            let current = shared[tid];
            shared[tid] = current + exp_val;
        }
    }

    sync_cube();

    // Tree reduction for sum
    tree_reduce_sum(shared, tid, block_size);

    sync_cube();

    // ---- Phase 3: Compute loss and logsumexp (thread 0 only) ----
    if tid == 0 {
        let lse = row_max + shared[0].ln();
        logsumexp_out[row] = lse;

        let label_idx = targets[row];
        let label_i32 = i32::cast_from(label_idx);

        if label_i32 == -100 {
            loss[row] = F::new(0.0);
        } else {
            let label_pos = u32::cast_from(label_idx) as usize;
            let raw = logits[base + label_pos * col_stride];
            let x = if do_softcap {
                cap * (raw / cap).tanh()
            } else {
                raw
            };
            loss[row] = lse - x;
        }
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

/// Fused cross-entropy backward: softmax + gradient + softcapping chain rule.
///
/// Each cube handles one row. Each thread independently computes gradients
/// for its slice of the vocab dimension — no shared memory or sync needed.
///
/// Gradient formula (without softcapping):
/// ```text
/// grad[i] = dloss * (softmax[i] - delta(i, label))
/// ```
///
/// With softcapping `sc(x) = cap * tanh(x / cap)`, chain rule:
/// ```text
/// d/dx sc(x) = 1 - tanh²(x / cap)
/// grad[i] = dloss * (softmax[sc(x)][i] - delta(i, label)) * (1 - tanh²(x / cap))
/// ```
///
/// The `logsumexp` saved from forward is computed on softcapped logits.
/// Backward re-applies softcapping to raw logits (from checkpoint) for softmax,
/// then multiplies by `sech²(x / cap)` for the chain rule.
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
    #[comptime] do_softcap: bool,
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

    // Gemma 2 2B final logit softcapping: 30.0 * tanh(x / 30.0)
    // Hardcoded as comptime constant — dead code eliminated when do_softcap=false.
    let cap = F::new(30.0);

    // Each thread strides over vocab elements — no sync needed between threads.
    let n_steps = vocab_size / block_size + 1;

    for step in 0..n_steps {
        let idx = tid + step * block_size;
        if idx < vocab_size {
            let raw = logits[base + idx * col_stride];

            // Apply softcapping and compute chain rule factor in one pass.
            // When do_softcap is false, dead code elimination removes all tanh ops.
            let t = if do_softcap {
                (raw / cap).tanh()
            } else {
                F::new(0.0) // unused, eliminated by compiler
            };
            let x_sc = if do_softcap { cap * t } else { raw };
            let chain_rule = if do_softcap {
                F::new(1.0) - t * t // sech²(x/cap) = 1 - tanh²(x/cap)
            } else {
                F::new(1.0)
            };

            // softmax(x_sc_i) = exp(x_sc_i - logsumexp)
            let softmax_val = (x_sc - lse).exp();

            // CE gradient: dloss * (softmax - delta(label))
            let is_label = idx == label_pos && !is_ignore;
            let delta = if is_label { F::new(1.0) } else { F::new(0.0) };

            let ce_grad = dloss * (softmax_val - delta);

            // Apply softcapping chain rule
            let grad = if do_softcap {
                ce_grad * chain_rule
            } else {
                ce_grad
            };

            grad_logits[base + idx * col_stride] = grad;
        }
    }
}

// ---------------------------------------------------------------------------
// Tree reduction helpers
// ---------------------------------------------------------------------------

/// Tree reduction for max in shared memory.
///
/// Performs log2(block_size) stages of pairwise max reduction.
/// Handles block_size up to 256 (8 stages). Larger block sizes
/// would need additional stages.
///
/// After this function + sync_cube(), the result is in shared[0].
#[cube]
fn tree_reduce_max<F: Float>(
    mut shared: SharedMemory<F>,
    tid: usize,
    #[comptime] block_size: usize,
) {
    if block_size >= 2 {
        let stride = block_size / 2;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 4 {
        let stride = block_size / 4;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 8 {
        let stride = block_size / 8;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 16 {
        let stride = block_size / 16;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 32 {
        let stride = block_size / 32;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 64 {
        let stride = block_size / 64;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 128 {
        let stride = block_size / 128;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
    if block_size >= 256 {
        let stride = block_size / 256;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = F::max(a, b);
        }
        sync_cube();
    }
}

/// Tree reduction for sum in shared memory.
///
/// Same structure as `tree_reduce_max` but using addition.
/// After this function + sync_cube(), the result is in shared[0].
#[cube]
fn tree_reduce_sum<F: Float>(
    mut shared: SharedMemory<F>,
    tid: usize,
    #[comptime] block_size: usize,
) {
    if block_size >= 2 {
        let stride = block_size / 2;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 4 {
        let stride = block_size / 4;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 8 {
        let stride = block_size / 8;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 16 {
        let stride = block_size / 16;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 32 {
        let stride = block_size / 32;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 64 {
        let stride = block_size / 64;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 128 {
        let stride = block_size / 128;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
    if block_size >= 256 {
        let stride = block_size / 256;
        if tid < stride {
            let a = shared[tid];
            let b = shared[tid + stride];
            shared[tid] = a + b;
        }
        sync_cube();
    }
}

// ---------------------------------------------------------------------------
// Host-side launch functions
// ---------------------------------------------------------------------------

/// Compute fused cross-entropy loss (forward) with optional logit softcapping.
///
/// # Arguments
/// * `logits`  — `[rows, vocab_size]` (must be contiguous, f32 recommended)
/// * `targets` — `[rows]` (integer tensor; -100 = ignore/padding)
/// * `softcap` — `None` for standard CE, `Some(cap)` to apply `cap * tanh(x / cap)`
///   before computing CE loss (Gemma 2 uses `Some(30.0)`)
///
/// # Returns
/// `(loss [rows], logsumexp [rows])` — logsumexp saved for backward.
pub fn fused_ce_forward<R: CubeRuntime>(
    logits: CubeTensor<R>,
    targets: CubeTensor<R>,
    softcap: Option<f32>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    // Save original dtype for casting loss back after kernel.
    let original_dtype = logits.dtype;

    // Cast to f32 for numerical stability. The logsumexp accumulation
    // sums exp() over the full vocab dimension (256K for Gemma 2), which
    // overflows f16 (max ~65504). Computing in f32 avoids this.
    let logits = into_contiguous(if original_dtype != DType::F32 {
        cast(logits, DType::F32)
    } else {
        logits
    });
    let targets = into_contiguous(targets);

    let [n_rows, _vocab_size] = logits.shape().dims::<2>();
    let client = logits.client.clone();
    let device = logits.device.clone();
    let f_dtype = logits.dtype; // Always f32 after cast above
    let i_dtype = targets.dtype;

    let loss_f32 = empty_device_dtype::<R>(
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

    let hw_max = client.properties().hardware.max_cube_dim.0 as usize;
    let block_size = CE_BLOCK_SIZE.min(hw_max).next_power_of_two();

    let do_softcap = softcap.is_some();

    let cube_count = CubeCount::Static(n_rows as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    fused_ce_forward_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        logits.into_tensor_arg(),
        targets.into_tensor_arg(),
        loss_f32.clone().into_tensor_arg(),
        logsumexp.clone().into_tensor_arg(),
        block_size,
        do_softcap,
        [f_dtype.into(), i_dtype.into()],
    );

    // Cast loss back to original dtype. Logsumexp stays f32 for backward.
    let loss = if original_dtype != DType::F32 {
        cast(loss_f32, original_dtype)
    } else {
        loss_f32
    };

    (loss, logsumexp)
}

/// Compute fused cross-entropy backward pass with optional softcapping chain rule.
///
/// # Arguments
/// * `logits`    — `[rows, vocab_size]` (raw logits, NOT softcapped)
/// * `grad_loss` — `[rows]`
/// * `logsumexp` — `[rows]` (saved from forward, computed on softcapped logits)
/// * `targets`   — `[rows]`
/// * `softcap`   — `None` for standard CE backward, `Some(cap)` to include
///   softcapping chain rule `1 - tanh²(x / cap)` in gradient
///
/// # Returns
/// Gradient w.r.t. logits `[rows, vocab_size]`
pub fn fused_ce_backward<R: CubeRuntime>(
    logits: CubeTensor<R>,
    grad_loss: CubeTensor<R>,
    logsumexp: CubeTensor<R>,
    targets: CubeTensor<R>,
    softcap: Option<f32>,
) -> CubeTensor<R> {
    // Save original dtype for casting grad_logits back after kernel.
    let original_dtype = logits.dtype;

    // Cast to f32 to match forward (logsumexp is f32 from forward).
    // Also avoids f16 precision loss in softmax gradient computation.
    let logits = into_contiguous(if original_dtype != DType::F32 {
        cast(logits, DType::F32)
    } else {
        logits
    });
    let grad_loss = into_contiguous(if original_dtype != DType::F32 {
        cast(grad_loss, DType::F32)
    } else {
        grad_loss
    });
    // logsumexp is already f32 from forward, just ensure contiguous.
    let logsumexp = into_contiguous(logsumexp);
    let targets = into_contiguous(targets);

    let [n_rows, vocab_size] = logits.shape().dims::<2>();
    let client = logits.client.clone();
    let device = logits.device.clone();
    let f_dtype = logits.dtype; // Always f32 after cast above
    let i_dtype = targets.dtype;

    let grad_logits_f32 = empty_device_dtype::<R>(
        client.clone(),
        device.clone(),
        Shape::new([n_rows, vocab_size]),
        f_dtype,
    );

    let hw_max = client.properties().hardware.max_cube_dim.0 as usize;
    let block_size = CE_BLOCK_SIZE.min(hw_max).next_power_of_two();

    let do_softcap = softcap.is_some();

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
        grad_logits_f32.clone().into_tensor_arg(),
        block_size,
        do_softcap,
        [f_dtype.into(), i_dtype.into()],
    );

    // Cast gradient back to original dtype for downstream ops.
    if original_dtype != DType::F32 {
        cast(grad_logits_f32, original_dtype)
    } else {
        grad_logits_f32
    }
}
