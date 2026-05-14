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

use crate::fused_ops::FusedCEBackend;
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
