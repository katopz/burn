//! Fused cross-entropy autodiff backward wrapper with optional logit softcapping.
//!
//! Implements [`FusedCEBackend`] for `Autodiff<B>` where `B: FusedCEBackend`.
//! Works with any inner backend — `CubeBackend`, `Fusion<CubeBackend>`, etc.
//!
//! # How it works
//!
//! The autodiff layer sits on top of the inner backend. For the fused CE loss:
//!
//! **Forward** (in autodiff graph):
//! 1. Extract inner tensor from autodiff-tracked logits: `logits.primitive`
//! 2. Call `B::ce_forward(logits.primitive, targets, softcap)` → `(loss_inner, logsumexp)`
//! 3. Save `(logits_checkpoint_id, logsumexp, targets, softcap)` as backward state
//! 4. Wrap `loss_inner` in a new autodiff node
//!
//! **Backward** (called during `loss.backward()` graph traversal):
//! 1. Retrieve original logits from checkpointer → `FloatTensor<B>`
//! 2. Consume upstream gradient → `FloatTensor<B>`
//! 3. Use saved `logsumexp` → `FloatTensor<B>` and `targets` → `IntTensor<B>`
//! 4. Call `B::ce_backward(logits, grad_loss, logsumexp, targets, softcap)` → `grad_logits`
//! 5. Register `grad_logits` for the logits parent node
//!
//! # Softcapping (Gemma 2)
//!
//! When `softcap` is `Some(cap)`:
//! - Forward: inner kernel applies `cap * tanh(x / cap)` inline
//! - Backward: inner kernel applies chain rule `1 - tanh²(x / cap)` to gradients
//! - The `softcap` value is saved as part of backward state
//!
//! # Why this is generic
//!
//! The actual kernel dispatch happens in the inner backend:
//! - `CubeBackend`: direct `#[cube]` kernel launch
//! - `Fusion<CubeBackend>`: fusion `Operation` that launches the kernel lazily
//!
//! This layer only manages the autodiff graph — it doesn't touch GPU kernels.

use crate::fused_ops::{FusedCEBackend, FusedLoraMLPBackend, FusedLoraMLPGrads, FusedLoraMLPState};
use burn::backend::autodiff::{
    Autodiff, NodeId,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::tensor::ops::FloatTensor;

/// Implement [`FusedCEBackend`] for `Autodiff<B, C>`.
///
/// Generic over any inner backend `B` that implements `FusedCEBackend`.
/// This covers both `Autodiff<CubeBackend<...>>` and `Autodiff<Fusion<CubeBackend<...>>>`.
impl<B, C> FusedCEBackend for Autodiff<B, C>
where
    B: FusedCEBackend + burn::tensor::backend::Backend,
    C: CheckpointStrategy,
{
    fn fused_ce_loss(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        /// Backward function for fused cross-entropy loss with optional softcapping.
        ///
        /// Implements `Backward<B, 1>`:
        /// - `B` is the inner backend (CubeBackend, Fusion<CubeBackend>, etc.)
        /// - `1` parent tensor (logits — targets is IntTensor, not autodiff-tracked)
        ///
        /// State saved during forward: `(logits_checkpoint_id, logsumexp, targets, softcap)`
        #[derive(Debug)]
        struct FusedCEBackward;

        impl<B: FusedCEBackend> Backward<B, 1> for FusedCEBackward {
            /// State needed for gradient computation.
            ///
            /// - `NodeId`: logits checkpoint ID — retrieve original logits via checkpointer
            /// - `FloatTensor<B>`: logsumexp — saved from forward, needed for
            ///   softmax computation: `softmax(x_i) = exp(x_i - logsumexp)`
            /// - `B::IntTensorPrimitive`: targets — class indices, -100 = ignore/padding
            /// - `Option<f32>`: softcap — `Some(cap)` to apply chain rule `1 - tanh²(x/cap)`
            type State = (NodeId, FloatTensor<B>, B::IntTensorPrimitive, Option<f32>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_logits] = ops.parents;
                let (logits_id, logsumexp, targets, softcap) = ops.state;

                // Retrieve original logits from the checkpointer.
                // These are the RAW logits (NOT softcapped) — the kernel re-applies
                // softcapping internally so it can compute the chain rule correctly.
                let logits: FloatTensor<B> = checkpointer.retrieve_node_output(logits_id);

                // Consume the upstream gradient for this operation's output.
                // The gradient flows from the loss → mask_fill → sum → div chain.
                let grad_loss: FloatTensor<B> = grads.consume::<B>(&ops.node);

                // Delegate to the inner backend's backward kernel.
                // For CubeBackend: direct #[cube] kernel launch
                // For Fusion<CubeBackend>: fusion Operation that launches lazily
                //
                // Computes: grad_logits[i,j] = dloss * (softmax[i,j] - delta(j, target))
                // When softcap is Some(cap): multiplies by (1 - tanh²(x/cap))
                let grad_logits: FloatTensor<B> =
                    B::ce_backward(logits, grad_loss, logsumexp, targets, softcap);

                // Register gradient for the logits parent node.
                // This feeds into the model's backward pass (matmul, attention, etc.)
                if let Some(node) = node_logits {
                    grads.register::<B>(node.id, grad_logits);
                }
            }
        }

        // Prepare autodiff tracking for the single parent tensor (logits).
        // Targets is IntTensor — it never enters the autodiff graph.
        match FusedCEBackward
            .prepare::<C>([logits.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // Checkpoint logits so the backward can retrieve the original tensor.
                let logits_id = prep.checkpoint(&logits);

                // Dispatch fused CE forward via the inner backend.
                // Returns both loss and logsumexp — we save logsumexp for backward.
                let (loss_inner, logsumexp) =
                    B::ce_forward(logits.primitive.clone(), targets.clone(), softcap);

                // Pack backward state: logits checkpoint + logsumexp + targets + softcap
                let state = (logits_id, logsumexp, targets, softcap);

                // Finish: wraps loss_inner in AutodiffTensor, registers
                // FusedCEBackward step in the autodiff graph.
                prep.finish(state, loss_inner)
            }
            OpsKind::UnTracked(prep) => {
                // No parent requires grad — compute forward only, discard logsumexp.
                let (loss_inner, _logsumexp) = B::ce_forward(logits.primitive, targets, softcap);
                prep.finish(loss_inner)
            }
        }
    }

    // ce_forward and ce_backward are NOT needed on Autodiff<B> because
    // the autodiff layer wraps operations, not exposes inner kernels.
    // These should never be called directly on an Autodiff backend.
    // We delegate to the inner backend for the actual kernel dispatch.

    fn ce_forward(
        _logits: FloatTensor<Self>,
        _targets: Self::IntTensorPrimitive,
        _softcap: Option<f32>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        unreachable!(
            "ce_forward should not be called on Autodiff<B> directly. \
             Use fused_ce_loss instead, which handles autodiff tracking."
        )
    }

    fn ce_backward(
        _logits: FloatTensor<Self>,
        _grad_loss: FloatTensor<Self>,
        _logsumexp: FloatTensor<Self>,
        _targets: Self::IntTensorPrimitive,
        _softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        unreachable!(
            "ce_backward should not be called on Autodiff<B> directly. \
             The backward is handled by FusedCEBackward during graph traversal."
        )
    }
}

// ---------------------------------------------------------------------------
// Fused LoRA MLP autodiff wrapper
// ---------------------------------------------------------------------------

/// Implement [`FusedLoraMLPBackend`] for `Autodiff<B, C>`.
///
/// Generic over any inner backend `B` that implements `FusedLoraMLPBackend`.
/// This covers both `Autodiff<CubeBackend<...>>` and `Autodiff<Fusion<CubeBackend<...>>>`.
///
/// # How it works
///
/// **Forward** (in autodiff graph):
/// 1. Extract inner tensors from autodiff-tracked inputs: `tensor.primitive`
/// 2. Call `B::fused_lora_mlp_forward_inner(...)` → `(output_inner, state)`
/// 3. Save `(7 checkpoint_ids, state.e, state.g, frozen_weights, scalars)` as backward state
/// 4. Wrap `output_inner` in a new autodiff node
///
/// **Backward** (called during `loss.backward()` graph traversal):
/// 1. Retrieve original inputs from checkpointer (7 tensors)
/// 2. Consume upstream gradient `dy`
/// 3. Call `B::fused_lora_mlp_backward(...)` → `FusedLoraMLPGrads`
/// 4. Register each gradient for the corresponding parent node

/// Saved state for the fused LoRA MLP backward pass.
///
/// Contains checkpoint IDs for the 7 trainable inputs, saved intermediates
/// from the forward pass (e, g), frozen base weights, and LoRA scaling factors.
/// Uses a named struct instead of a tuple because Rust only implements [`Debug`]
/// for tuples up to 12 elements (we have 15).
#[derive(Clone, Debug)]
struct FusedLoraMLPBackwardState<B: burn::tensor::backend::Backend> {
    /// Checkpoint IDs for the 7 trainable inputs.
    x_id: NodeId,
    gate_a_id: NodeId,
    gate_b_id: NodeId,
    up_a_id: NodeId,
    up_b_id: NodeId,
    down_a_id: NodeId,
    down_b_id: NodeId,
    /// Saved intermediates from forward.
    e: FloatTensor<B>,
    g: FloatTensor<B>,
    /// Frozen base weights (not autodiff-tracked).
    gate_w: FloatTensor<B>,
    up_w: FloatTensor<B>,
    down_w: FloatTensor<B>,
    /// LoRA scaling factors.
    gate_s: f64,
    up_s: f64,
    down_s: f64,
}

impl<B, C> FusedLoraMLPBackend for Autodiff<B, C>
where
    B: FusedLoraMLPBackend + burn::tensor::backend::Backend,
    C: CheckpointStrategy,
{
    fn fused_lora_mlp_forward(
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
    ) -> FloatTensor<Self> {
        /// Backward function for fused LoRA MLP.
        ///
        /// Implements `Backward<B, 7>`:
        /// - `B` is the inner backend (CubeBackend, Fusion<CubeBackend>, etc.)
        /// - `7` parent tensors: x, gate_a, gate_b, up_a, up_b, down_a, down_b
        ///
        /// Frozen base weights (gate_w, up_w, down_w) and scalars are saved in state,
        /// not tracked as parents since they don't require gradients.
        #[derive(Debug)]
        struct FusedLoraMLPBackward;

        impl<B: FusedLoraMLPBackend> Backward<B, 7> for FusedLoraMLPBackward {
            /// State needed for gradient computation.
            ///
            /// - 7 `NodeId`s: checkpoint IDs for each trainable input
            /// - `FloatTensor<B>`: e (gate pre-activation) and g (up projection) from forward
            /// - `FloatTensor<B>`: frozen base weights (gate_w, up_w, down_w)
            /// - `f64`: LoRA scaling factors (gate_s, up_s, down_s)
            type State = FusedLoraMLPBackwardState<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_ga, node_gb, node_ua, node_ub, node_da, node_db] = ops.parents;
                let FusedLoraMLPBackwardState {
                    x_id,
                    gate_a_id,
                    gate_b_id,
                    up_a_id,
                    up_b_id,
                    down_a_id,
                    down_b_id,
                    e,
                    g,
                    gate_w,
                    up_w,
                    down_w,
                    gate_s,
                    up_s,
                    down_s,
                } = ops.state;

                // Retrieve original inputs from the checkpointer.
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_id);
                let gate_a: FloatTensor<B> = checkpointer.retrieve_node_output(gate_a_id);
                let gate_b: FloatTensor<B> = checkpointer.retrieve_node_output(gate_b_id);
                let up_a: FloatTensor<B> = checkpointer.retrieve_node_output(up_a_id);
                let up_b: FloatTensor<B> = checkpointer.retrieve_node_output(up_b_id);
                let down_a: FloatTensor<B> = checkpointer.retrieve_node_output(down_a_id);
                let down_b: FloatTensor<B> = checkpointer.retrieve_node_output(down_b_id);

                // Consume the upstream gradient for this operation's output.
                let dy: FloatTensor<B> = grads.consume::<B>(&ops.node);

                // Delegate to the inner backend's backward kernel.
                let grad_results: FusedLoraMLPGrads<B> = B::fused_lora_mlp_backward(
                    x,
                    gate_a,
                    gate_b,
                    gate_s,
                    up_a,
                    up_b,
                    up_s,
                    down_a,
                    down_b,
                    down_s,
                    gate_w,
                    up_w,
                    down_w,
                    FusedLoraMLPState { e, g },
                    dy,
                );

                // Register gradients for each parent node that requires grad.
                if let Some(node) = node_x {
                    grads.register::<B>(node.id, grad_results.dx);
                }
                if let Some(node) = node_ga {
                    grads.register::<B>(node.id, grad_results.d_gate_a);
                }
                if let Some(node) = node_gb {
                    grads.register::<B>(node.id, grad_results.d_gate_b);
                }
                if let Some(node) = node_ua {
                    grads.register::<B>(node.id, grad_results.d_up_a);
                }
                if let Some(node) = node_ub {
                    grads.register::<B>(node.id, grad_results.d_up_b);
                }
                if let Some(node) = node_da {
                    grads.register::<B>(node.id, grad_results.d_down_a);
                }
                if let Some(node) = node_db {
                    grads.register::<B>(node.id, grad_results.d_down_b);
                }
            }
        }

        // Prepare autodiff tracking for the 7 trainable parent tensors.
        // Frozen base weights (gate_w, up_w, down_w) are NOT autodiff-tracked.
        match FusedLoraMLPBackward
            .prepare::<C>([
                x.node.clone(),
                gate_a.node.clone(),
                gate_b.node.clone(),
                up_a.node.clone(),
                up_b.node.clone(),
                down_a.node.clone(),
                down_b.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                // Checkpoint all 7 trainable inputs for backward retrieval.
                let x_id = prep.checkpoint(&x);
                let gate_a_id = prep.checkpoint(&gate_a);
                let gate_b_id = prep.checkpoint(&gate_b);
                let up_a_id = prep.checkpoint(&up_a);
                let up_b_id = prep.checkpoint(&up_b);
                let down_a_id = prep.checkpoint(&down_a);
                let down_b_id = prep.checkpoint(&down_b);

                // Dispatch fused LoRA MLP forward via the inner backend.
                let (out_inner, state) = B::fused_lora_mlp_forward_inner(
                    x.primitive.clone(),
                    gate_a.primitive.clone(),
                    gate_b.primitive.clone(),
                    gate_s,
                    up_a.primitive.clone(),
                    up_b.primitive.clone(),
                    up_s,
                    down_a.primitive.clone(),
                    down_b.primitive.clone(),
                    down_s,
                    gate_w.primitive.clone(),
                    up_w.primitive.clone(),
                    down_w.primitive.clone(),
                );

                // Pack backward state
                let backward_state = FusedLoraMLPBackwardState {
                    x_id,
                    gate_a_id,
                    gate_b_id,
                    up_a_id,
                    up_b_id,
                    down_a_id,
                    down_b_id,
                    e: state.e,
                    g: state.g,
                    gate_w: gate_w.primitive,
                    up_w: up_w.primitive,
                    down_w: down_w.primitive,
                    gate_s,
                    up_s,
                    down_s,
                };

                prep.finish(backward_state, out_inner)
            }
            OpsKind::UnTracked(prep) => {
                // No parent requires grad — compute forward only, discard state.
                let (out_inner, _state) = B::fused_lora_mlp_forward_inner(
                    x.primitive,
                    gate_a.primitive,
                    gate_b.primitive,
                    gate_s,
                    up_a.primitive,
                    up_b.primitive,
                    up_s,
                    down_a.primitive,
                    down_b.primitive,
                    down_s,
                    gate_w.primitive,
                    up_w.primitive,
                    down_w.primitive,
                );
                prep.finish(out_inner)
            }
        }
    }

    fn fused_lora_mlp_forward_inner(
        _x: FloatTensor<Self>,
        _gate_a: FloatTensor<Self>,
        _gate_b: FloatTensor<Self>,
        _gate_s: f64,
        _up_a: FloatTensor<Self>,
        _up_b: FloatTensor<Self>,
        _up_s: f64,
        _down_a: FloatTensor<Self>,
        _down_b: FloatTensor<Self>,
        _down_s: f64,
        _gate_w: FloatTensor<Self>,
        _up_w: FloatTensor<Self>,
        _down_w: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FusedLoraMLPState<Self>) {
        unreachable!(
            "fused_lora_mlp_forward_inner should not be called on Autodiff<B> directly. \
             Use fused_lora_mlp_forward instead, which handles autodiff tracking."
        )
    }

    fn fused_lora_mlp_backward(
        _x: FloatTensor<Self>,
        _gate_a: FloatTensor<Self>,
        _gate_b: FloatTensor<Self>,
        _gate_s: f64,
        _up_a: FloatTensor<Self>,
        _up_b: FloatTensor<Self>,
        _up_s: f64,
        _down_a: FloatTensor<Self>,
        _down_b: FloatTensor<Self>,
        _down_s: f64,
        _gate_w: FloatTensor<Self>,
        _up_w: FloatTensor<Self>,
        _down_w: FloatTensor<Self>,
        _state: FusedLoraMLPState<Self>,
        _dy: FloatTensor<Self>,
    ) -> FusedLoraMLPGrads<Self> {
        unreachable!(
            "fused_lora_mlp_backward should not be called on Autodiff<B> directly. \
             The backward is handled by FusedLoraMLPBackward during graph traversal."
        )
    }
}
