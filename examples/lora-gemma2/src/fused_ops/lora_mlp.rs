//! Fused LoRA MLP forward/backward for GeGLU-based MLP blocks (Gemma 2).
//!
//! Fuses gate+up+down LoRA projections with GeGLU activation into
//! a single autodiff backward step, replacing 15+ individual backward steps
//! per MLP block.
//!
//! # What It Replaces
//!
//! Standard path (3 × LoraLinear, each with autodiff tracking):
//! ```text
//! gate = gelu(X @ (G + Ag@Bg*sg))        → 3 autodiff ops (matmul + lora matmul + gelu)
//! up   = X @ (U + Au@Bu*su)              → 2 autodiff ops (matmul + lora matmul)
//! down = (gate*up) @ (W + Aw@Bw*sw)      → 3 autodiff ops (mul + matmul + lora matmul)
//! Total: ~15 backward steps (× 26 blocks = ~390 steps)
//! ```
//!
//! Fused path:
//! ```text
//! 1 forward dispatch + 1 backward dispatch → 2 total
//! ```
//!
//! # Architecture
//!
//! Uses the same two-layer trait design as [`FusedCEBackend`](crate::fused_ops::FusedCEBackend):
//! - [`FusedLoraMLPBackend`] — user-facing trait with `fused_lora_mlp_forward` +
//!   internal `fused_lora_mlp_forward_inner`/`fused_lora_mlp_backward` for autodiff integration
//! - Implementations:
//!   - `CubeBackend` — direct kernel dispatch
//!   - `Fusion<CubeBackend>` — fusion operation dispatch
//!   - `Autodiff<B>` — autodiff wrapping (uses inner B's forward+backward)
//!
//! # GeGLU Activation
//!
//! Gemma 2 uses GeGLU: `GeGLU(x) = gelu(x @ G) * (x @ U)`.
//! The fused forward computes:
//! ```text
//! e = X @ (G + Ag@Bg*sg)           // gate pre-activation
//! g = X @ (U + Au@Bu*su)           // up projection
//! out = gelu(e) * g @ (D + Ad@Bd*sd)  // down projection
//! ```
//!
//! The backward needs `e` (for GELU derivative) and `g` (for element-wise mul),
//! saved in [`FusedLoraMLPState`].
//!
//! # LoRA Scaling
//!
//! Each LoRA pair `(A, B)` has a scaling factor `s`:
//! - Effective weight: `W + A @ B * s`
//! - `s = alpha / rank` (typically `16 / 8 = 2.0`)

use burn::tensor::{Tensor, TensorPrimitive, backend::Backend, ops::FloatTensor};

// ---------------------------------------------------------------------------
// State structs
// ---------------------------------------------------------------------------

/// Saved intermediate values from fused LoRA MLP forward for backward.
///
/// These are the minimum values needed to compute all gradients in the backward
/// pass without recomputing any forward operations.
///
/// # Fields
///
/// - `e`: Gate pre-activation `X @ (G + Ag @ Bg * sg)` — needed for GELU derivative
///   `gelu'(e) = 0.5 * (1 + erf(e / sqrt(2))) + e * pdf(e)`
/// - `g`: Up projection output `X @ (U + Au @ Bu * su)` — needed for element-wise
///   multiply backward: `d(gelu(e) * g) = d_gelu * g + gelu(e) * d_g`
pub struct FusedLoraMLPState<B: Backend> {
    /// Gate pre-activation: `e = X @ (G + Ag @ Bg * sg)` — needed for GELU derivative.
    pub e: FloatTensor<B>,
    /// Up projection output: `g = X @ (U + Au @ Bu * su)` — needed for GeGLU backward.
    pub g: FloatTensor<B>,
}

/// All gradients computed in the fused LoRA MLP backward.
///
/// Contains gradients for the input and all six LoRA weight matrices
/// (A and B for gate, up, and down projections). Base weights are frozen
/// and do not receive gradients.
///
/// # Gradient Shapes
///
/// | Gradient | Shape | Notes |
/// |----------|-------|-------|
/// | `dx` | `[N, d_in]` | Input gradient for upstream layers |
/// | `d_gate_a` | `[d_in, r]` | Gate LoRA A |
/// | `d_gate_b` | `[r, d_mid]` | Gate LoRA B |
/// | `d_up_a` | `[d_in, r]` | Up LoRA A |
/// | `d_up_b` | `[r, d_mid]` | Up LoRA B |
/// | `d_down_a` | `[d_mid, r]` | Down LoRA A |
/// | `d_down_b` | `[r, d_out]` | Down LoRA B |
pub struct FusedLoraMLPGrads<B: Backend> {
    /// dX: input gradient `[N, d_in]`.
    pub dx: FloatTensor<B>,
    /// dAg: gate LoRA A gradient `[d_in, r]`.
    pub d_gate_a: FloatTensor<B>,
    /// dBg: gate LoRA B gradient `[r, d_mid]`.
    pub d_gate_b: FloatTensor<B>,
    /// dAu: up LoRA A gradient `[d_in, r]`.
    pub d_up_a: FloatTensor<B>,
    /// dBu: up LoRA B gradient `[r, d_mid]`.
    pub d_up_b: FloatTensor<B>,
    /// dAw: down LoRA A gradient `[d_mid, r]`.
    pub d_down_a: FloatTensor<B>,
    /// dBw: down LoRA B gradient `[r, d_out]`.
    pub d_down_b: FloatTensor<B>,
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Backend trait for fused LoRA MLP forward/backward.
///
/// Extends burn's [`Backend`] with fused GeGLU MLP operations that combine
/// gate, up, and down LoRA projections into a single autodiff step.
///
/// # Implementations
///
/// | Type | Forward | Backward |
/// |------|---------|----------|
/// | `CubeBackend` | Direct `#[cube]` kernel | Direct `#[cube]` kernel |
/// | `Fusion<CubeBackend>` | Fusion `Operation` | Fusion `Operation` |
/// | `Autodiff<B>` | Autodiff graph node | `Backward<B, 7>` step |
///
/// # Trait Methods
///
/// - [`fused_lora_mlp_forward`](Self::fused_lora_mlp_forward) — user-facing,
///   returns output only
/// - [`fused_lora_mlp_forward_inner`](Self::fused_lora_mlp_forward_inner) —
///   internal, returns `(output, state)` for autodiff state saving
/// - [`fused_lora_mlp_backward`](Self::fused_lora_mlp_backward) — internal,
///   computes all gradients during graph traversal
pub trait FusedLoraMLPBackend: Backend {
    // -----------------------------------------------------------------------
    // User-facing API
    // -----------------------------------------------------------------------

    /// Compute fused LoRA MLP forward, returning only the output tensor.
    ///
    /// Fuses gate+up+down LoRA projections with GeGLU activation:
    /// ```text
    /// e = X @ (G + Ag@Bg*sg)
    /// out = gelu(e) * (X @ (U + Au@Bu*su)) @ (D + Ad@Bd*sd)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` — Input activations `[N, d_in]`
    /// * `gate_a`, `gate_b`, `gate_s` — Gate LoRA: A `[d_in, r]`, B `[r, d_mid]`, scale
    /// * `up_a`, `up_b`, `up_s` — Up LoRA: A `[d_in, r]`, B `[r, d_mid]`, scale
    /// * `down_a`, `down_b`, `down_s` — Down LoRA: A `[d_mid, r]`, B `[r, d_out]`, scale
    /// * `gate_w` — Frozen gate base weight `[d_in, d_mid]`
    /// * `up_w` — Frozen up base weight `[d_in, d_mid]`
    /// * `down_w` — Frozen down base weight `[d_mid, d_out]`
    ///
    /// # Returns
    ///
    /// Output tensor `[N, d_out]`
    fn fused_lora_mlp_forward(
        x: FloatTensor<Self>,
        // LoRA weights (trainable)
        gate_a: FloatTensor<Self>,
        gate_b: FloatTensor<Self>,
        gate_s: f64,
        up_a: FloatTensor<Self>,
        up_b: FloatTensor<Self>,
        up_s: f64,
        down_a: FloatTensor<Self>,
        down_b: FloatTensor<Self>,
        down_s: f64,
        // Frozen base weights
        gate_w: FloatTensor<Self>,
        up_w: FloatTensor<Self>,
        down_w: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    // -----------------------------------------------------------------------
    // Internal API (for autodiff backward state management)
    // -----------------------------------------------------------------------

    /// Forward kernel returning both output and saved state.
    ///
    /// Used by `Autodiff<B>` to save intermediates for the backward pass.
    /// Non-autodiff backends should delegate `fused_lora_mlp_forward` to this
    /// and discard the state.
    ///
    /// # Returns
    ///
    /// `(output [N, d_out], FusedLoraMLPState)`
    fn fused_lora_mlp_forward_inner(
        x: FloatTensor<Self>,
        gate_a: FloatTensor<Self>,
        gate_b: FloatTensor<Self>,
        gate_s: f64,
        up_a: FloatTensor<Self>,
        up_b: FloatTensor<Self>,
        up_s: f64,
        down_a: FloatTensor<Self>,
        down_b: FloatTensor<Self>,
        down_s: f64,
        gate_w: FloatTensor<Self>,
        up_w: FloatTensor<Self>,
        down_w: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FusedLoraMLPState<Self>);

    /// Backward kernel computing all gradients for fused LoRA MLP.
    ///
    /// Called by `Autodiff<B>`'s `Backward<B, 7>` implementation during
    /// graph traversal. Computes gradients for:
    /// - Input `x` (for upstream layers)
    /// - All 6 LoRA weight matrices (gate, up, down × A, B)
    ///
    /// Base weights are frozen and do not receive gradients.
    ///
    /// # Arguments
    ///
    /// * `x` — Original input `[N, d_in]`
    /// * `gate_a`, `gate_b`, `gate_s` — Gate LoRA weights + scale
    /// * `up_a`, `up_b`, `up_s` — Up LoRA weights + scale
    /// * `down_a`, `down_b`, `down_s` — Down LoRA weights + scale
    /// * `gate_w`, `up_w`, `down_w` — Frozen base weights
    /// * `state` — Saved intermediates from forward (`e`, `g`)
    /// * `dy` — Upstream gradient `[N, d_out]`
    ///
    /// # Returns
    ///
    /// [`FusedLoraMLPGrads`] containing all computed gradients.
    fn fused_lora_mlp_backward(
        x: FloatTensor<Self>,
        gate_a: FloatTensor<Self>,
        gate_b: FloatTensor<Self>,
        gate_s: f64,
        up_a: FloatTensor<Self>,
        up_b: FloatTensor<Self>,
        up_s: f64,
        down_a: FloatTensor<Self>,
        down_b: FloatTensor<Self>,
        down_s: f64,
        gate_w: FloatTensor<Self>,
        up_w: FloatTensor<Self>,
        down_w: FloatTensor<Self>,
        // Saved intermediates
        state: FusedLoraMLPState<Self>,
        // Upstream gradient
        dy: FloatTensor<Self>,
    ) -> FusedLoraMLPGrads<Self>;
}

// ---------------------------------------------------------------------------
// Public free function
// ---------------------------------------------------------------------------

/// Compute fused LoRA MLP with autodiff support.
///
/// Replaces the standard chain of 3 LoraLinear modules + GeGLU activation
/// (15+ autodiff steps) with a single fused forward + backward dispatch.
///
/// See [`FusedLoraMLPBackend`] for architectural details.
///
/// # Arguments
///
/// * `x` — Input activations `[N, d_in]`
/// * `gate_a`, `gate_b`, `gate_s` — Gate LoRA: A `[d_in, r]`, B `[r, d_mid]`, scale
/// * `up_a`, `up_b`, `up_s` — Up LoRA: A `[d_in, r]`, B `[r, d_mid]`, scale
/// * `down_a`, `down_b`, `down_s` — Down LoRA: A `[d_mid, r]`, B `[r, d_out]`, scale
/// * `gate_w` — Frozen gate base weight `[d_in, d_mid]`
/// * `up_w` — Frozen up base weight `[d_in, d_mid]`
/// * `down_w` — Frozen down base weight `[d_mid, d_out]`
///
/// # Returns
///
/// Output tensor `[N, d_out]` — autodiff-tracked for gradient computation.
///
/// # Example (in Gemma 2 MLP block)
///
/// ```ignore
/// let output = fused_lora_mlp::<B>(
///     x,
///     // Gate LoRA (trainable)
///     self.gate_lora_a.weight(), self.gate_lora_b.weight(), self.gate_lora_s,
///     // Up LoRA (trainable)
///     self.up_lora_a.weight(), self.up_lora_b.weight(), self.up_lora_s,
///     // Down LoRA (trainable)
///     self.down_lora_a.weight(), self.down_lora_b.weight(), self.down_lora_s,
///     // Frozen base weights
///     self.gate_weight.val(),
///     self.up_weight.val(),
///     self.down_weight.val(),
/// );
/// ```
pub fn fused_lora_mlp<B: FusedLoraMLPBackend>(
    x: Tensor<B, 2>,
    // LoRA weights
    gate_a: Tensor<B, 2>,
    gate_b: Tensor<B, 2>,
    gate_s: f64,
    up_a: Tensor<B, 2>,
    up_b: Tensor<B, 2>,
    up_s: f64,
    down_a: Tensor<B, 2>,
    down_b: Tensor<B, 2>,
    down_s: f64,
    // Frozen base weights
    gate_w: Tensor<B, 2>,
    up_w: Tensor<B, 2>,
    down_w: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let output = B::fused_lora_mlp_forward(
        x.into_primitive().tensor(),
        gate_a.into_primitive().tensor(),
        gate_b.into_primitive().tensor(),
        gate_s,
        up_a.into_primitive().tensor(),
        up_b.into_primitive().tensor(),
        up_s,
        down_a.into_primitive().tensor(),
        down_b.into_primitive().tensor(),
        down_s,
        gate_w.into_primitive().tensor(),
        up_w.into_primitive().tensor(),
        down_w.into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}
