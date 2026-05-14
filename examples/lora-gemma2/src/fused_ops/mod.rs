//! Fused cross-entropy loss with autodiff support and optional logit softcapping.
//!
//! Replaces `log_softmax → gather → neg` (10+ kernel launches) with a single
//! fused GPU kernel, avoiding materialization of `[batch*seq, 256K]` log_probs.
//!
//! When `softcap` is `Some(cap)`, also fuses Gemma 2's logit softcapping
//! `cap * tanh(x / cap)` into the kernel — replacing 4 additional ops
//! (cast, div_scalar, tanh, mul_scalar) and avoiding materialization of
//! the softcapped logits tensor.
//!
//! # Architecture
//!
//! Uses a two-layer trait design:
//! - [`FusedCEBackend`] — user-facing trait with `fused_ce_loss` + internal
//!   `ce_forward`/`ce_backward` for autodiff integration
//! - Implementations:
//!   - `CubeBackend` — direct kernel dispatch
//!   - `Fusion<CubeBackend>` — fusion operation dispatch
//!   - `Autodiff<B>` — autodiff wrapping (uses inner B's forward+backward)
//!
//! # What It Replaces
//!
//! Standard burn CE loss chain (per training step):
//! ```text
//! cast_f32 → div_scalar(cap) → tanh → mul_scalar(cap) → cast →
//! log_softmax → gather → neg → mask_fill → sum → div
//! (~15 autodiff steps × GPU kernel launches, materializes [batch*seq, 256K] twice)
//! ```
//!
//! Fused approach:
//! ```text
//! fused_ce_loss (1 GPU kernel with inline softcapping) → mask_fill → sum → div
//! fused_ce_backward (1 GPU kernel with chain rule, replaces ~15 backward steps)
//! ```
//!
//! The mask/sum/div still go through standard autodiff on the small `[rows]`
//! loss tensor — this is fine because those ops are cheap.
//!
//! # Softcapping (Gemma 2)
//!
//! Gemma 2 applies `sc(x) = cap * tanh(x / cap)` to logits before CE loss.
//! With `softcap = Some(30.0)`:
//! - Forward: computes logsumexp and loss on softcapped values inline
//! - Backward: applies chain rule `d/dx sc(x) = 1 - tanh²(x/cap)` to gradients
//!
//! # Reference
//! - `unsloth/kernels/cross_entropy_loss.py` — Triton kernel adapted from
//! - `burn/examples/custom-cubecl-kernel/` — Backend trait extension pattern

mod backward;
mod forward;

use burn::tensor::{Int, Tensor, TensorPrimitive, ops::FloatTensor};

/// Backend trait for fused cross-entropy loss operations with optional softcapping.
///
/// Extends burn's [`Backend`](burn::tensor::backend::Backend) with:
/// 1. **User-facing** `fused_ce_loss` — returns per-token loss (single tensor)
/// 2. **Internal** `ce_forward` / `ce_backward` — used by the autodiff wrapper
///    to save/restore logsumexp for gradient computation
///
/// # Implementations
///
/// | Type | Forward | Backward |
/// |------|---------|----------|
/// | `CubeBackend` | Direct `#[cube]` kernel | Direct `#[cube]` kernel |
/// | `Fusion<CubeBackend>` | Fusion `Operation` | Fusion `Operation` |
/// | `Autodiff<B>` | Autodiff graph node | `Backward<B, 1>` step |
pub trait FusedCEBackend: burn::tensor::backend::Backend {
    // -----------------------------------------------------------------------
    // User-facing API
    // -----------------------------------------------------------------------

    /// Compute fused cross-entropy loss per token with optional logit softcapping.
    ///
    /// Fuses `log_softmax + gather + neg` into a single GPU kernel dispatch.
    /// When `softcap` is `Some(cap)`, also fuses `cap * tanh(x / cap)` inline.
    ///
    /// # Arguments
    /// * `logits`  — `[rows, vocab_size]` (f32 recommended for numerical stability)
    /// * `targets` — `[rows]` (int tensor, -100 = ignore/padding)
    /// * `softcap` — `None` for standard CE, `Some(cap)` for Gemma 2 softcapping
    ///
    /// # Returns
    /// Per-token losses `[rows]`:
    /// - `loss[i] = logsumexp(sc(logits[i,:])) - sc(logits[i, targets[i]])`
    /// - `loss[i] = 0.0` when `targets[i] == -100` (padding/ignore)
    /// - where `sc(x) = cap * tanh(x / cap)` when softcap is `Some(cap)`,
    ///   or `sc(x) = x` when softcap is `None`
    ///
    /// For `Autodiff<B>`, the result is tracked in the autodiff graph with a
    /// custom backward that calls [`ce_backward`](Self::ce_backward).
    fn fused_ce_loss(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self>;

    // -----------------------------------------------------------------------
    // Internal API (for autodiff backward state management)
    // -----------------------------------------------------------------------

    /// Forward kernel returning both loss and logsumexp.
    ///
    /// Used by `Autodiff<B>` to save logsumexp as backward state.
    /// Non-autodiff backends should delegate `fused_ce_loss` to this and
    /// discard logsumexp.
    ///
    /// # Returns
    /// `(per_token_loss [rows], logsumexp [rows])`
    fn ce_forward(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// Backward kernel computing `d(loss)/d(logits)` with optional softcapping.
    ///
    /// Used by `Autodiff<B>`'s `Backward<B, 1>` implementation.
    /// Single GPU kernel: recomputes softmax via saved logsumexp, then
    /// computes gradient as `dloss * (softmax[i] - delta(i, target))`.
    /// When softcap is `Some(cap)`, multiplies by chain rule `(1 - tanh²(x/cap))`.
    ///
    /// # Arguments
    /// * `logits`    — Original logits `[rows, vocab_size]` (raw, NOT softcapped)
    /// * `grad_loss` — Upstream gradient `[rows]`
    /// * `logsumexp` — Saved from forward `[rows]` (computed on softcapped logits)
    /// * `targets`   — Target class indices `[rows]`
    /// * `softcap`   — `None` for standard CE backward, `Some(cap)` for chain rule
    ///
    /// # Returns
    /// Gradient w.r.t. logits `[rows, vocab_size]`
    fn ce_backward(
        logits: FloatTensor<Self>,
        grad_loss: FloatTensor<Self>,
        logsumexp: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self>;
}

/// Compute fused cross-entropy loss with autodiff support and optional softcapping.
///
/// Replaces the standard chain: `softcap → log_softmax → gather → neg`
/// (15+ kernel launches) with a single fused kernel dispatch.
/// When used with an autodiff backend, the backward pass also uses a single
/// fused kernel with the softcapping chain rule built in.
///
/// # Arguments
/// * `logits`  — `[rows, vocab_size]` (f32 recommended for numerical stability)
/// * `targets` — `[rows]` (int tensor, -100 = ignore/padding)
/// * `softcap` — `None` for standard CE, `Some(cap)` to fuse
///   `cap * tanh(x / cap)` inline (Gemma 2 uses `Some(30.0)`)
///
/// # Returns
/// Per-token losses `[rows]` — autodiff-tracked for gradient computation.
///
/// # Example (in TrainStep)
/// ```ignore
/// // Gemma 2: softcapping fused into CE kernel
/// let token_losses = fused_ce_loss::<B>(logits_flat, targets_flat, Some(30.0));
///
/// // Standard CE (no softcapping)
/// let token_losses = fused_ce_loss::<B>(logits_flat, targets_flat, None);
/// ```
pub fn fused_ce_loss<B: FusedCEBackend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    softcap: Option<f32>,
) -> Tensor<B, 1> {
    let loss = B::fused_ce_loss(
        logits.into_primitive().tensor(),
        targets.into_primitive(),
        softcap,
    );
    Tensor::from_primitive(TensorPrimitive::Float(loss))
}
