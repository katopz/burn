//! Fused CE backend implementations for CubeBackend and Fusion<CubeBackend>.
//!
//! Two implementation paths:
//! - **CubeBackend**: Direct `#[cube]` kernel dispatch (used in tests or bare backend)
//! - **Fusion<CubeBackend>**: Fusion `Operation` dispatch (what Metal/Wgpu actually use)
//!
//! Both pass `softcap: Option<f32>` to the kernel launch functions for
//! optional inline logit softcapping (Gemma 2).

use crate::fused_ops::FusedCEBackend;
use crate::kernel::cross_entropy::{fused_ce_backward, fused_ce_forward};
use burn::tensor::ops::FloatTensor;
use burn_backend::{Shape, TensorMetadata};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};
use burn_cubecl_fusion::CubeFusionHandle;
use burn_ir::HandleContainer;
use core::marker::PhantomData;

// ---------------------------------------------------------------------------
// Direct CubeBackend impl (no Fusion wrapper)
// ---------------------------------------------------------------------------

/// Implement fused CE backend trait for `CubeBackend` (no Fusion wrapper).
///
/// Directly dispatches the `#[cube]` kernels — no intermediate tensor ops.
/// `FloatTensor<CubeBackend<R, F, I, BT>> = CubeTensor<R>`
/// `IntTensorPrimitive = CubeTensor<R>` — same type, works with kernel signature.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusedCEBackend
    for CubeBackend<R, F, I, BT>
{
    fn fused_ce_loss(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        let (loss, _logsumexp) = Self::ce_forward(logits, targets, softcap);
        loss
    }

    fn ce_forward(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        fused_ce_forward(logits, targets, softcap)
    }

    fn ce_backward(
        logits: FloatTensor<Self>,
        grad_loss: FloatTensor<Self>,
        logsumexp: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        fused_ce_backward(logits, grad_loss, logsumexp, targets, softcap)
    }
}

// ---------------------------------------------------------------------------
// Fusion<CubeBackend> impl — Metal/Wgpu use Fusion<CubeBackend<...>>
// ---------------------------------------------------------------------------

/// Implement [`FusedCEBackend`] for `Fusion<CubeBackend<...>>`.
///
/// This is the actual backend type used when `Metal` or `Wgpu` is selected,
/// since `Metal = Wgpu = Fusion<CubeBackend<...>>`.
///
/// Uses the fusion [`Operation`] pattern to integrate our custom kernels into
/// the fusion stream. When the stream flushes, each operation extracts
/// [`CubeTensor`]s from the fusion handle container, calls the fused kernel,
/// and registers the result — no intermediate tensor materialization.
///
/// # Why [`CustomOpIr`]
///
/// The fusion optimizer uses [`OperationIr`] to decide which ops to fuse.
/// Using `Custom` prevents the optimizer from fusing our kernel with neighbors
/// (which would be incorrect since our kernel does something completely different).
/// Our kernel is already fused internally — we don't want external fusion.
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm",
))]
impl<R, F, I, BT> FusedCEBackend for burn_fusion::Fusion<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_ce_loss(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        let (loss, _logsumexp) = Self::ce_forward(logits, targets, softcap);
        loss
    }

    fn ce_forward(
        logits: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        use burn_cubecl::fusion::FusionCubeRuntime;
        use burn_fusion::stream::{Operation, OperationStreams};
        use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorStatus};

        // -------------------------------------------------------------------
        // Fusion Operation: forward kernel producing 2 outputs (loss + logsumexp)
        // -------------------------------------------------------------------
        #[derive(Debug)]
        struct FusedCEForwardOp<R, F, I, BT> {
            logits_ir: burn_ir::TensorIr,
            targets_ir: burn_ir::TensorIr,
            loss_id: burn_ir::TensorId,
            logsumexp_id: burn_ir::TensorId,
            softcap: Option<f32>,
            _phantom: PhantomData<(R, F, I, BT)>,
        }

        impl<R, F, I, BT> Operation<FusionCubeRuntime<R>> for FusedCEForwardOp<R, F, I, BT>
        where
            R: CubeRuntime,
            F: FloatElement,
            I: IntElement,
            BT: BoolElement,
        {
            fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<R>>) {
                // Resolve FusionTensor IR → actual CubeTensors
                let logits: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.logits_ir);
                let targets: CubeTensor<R> =
                    handles.get_int_tensor::<CubeBackend<R, F, I, BT>>(&self.targets_ir);

                // Single fused GPU kernel: softcap + logsumexp + CE loss
                let (loss, logsumexp) = fused_ce_forward(logits, targets, self.softcap);

                // Register both outputs in the handle container
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(&self.loss_id, loss);
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.logsumexp_id,
                    logsumexp,
                );
            }
        }

        // -------------------------------------------------------------------
        // Setup: register inputs/outputs with fusion stream
        // -------------------------------------------------------------------
        let client = logits.client.clone();
        let out_dtype = logits.dtype;

        // Output shapes: both [rows] from logits [rows, vocab]
        let n_rows = logits.shape().dims::<2>()[0];
        let out_shape = Shape::from(vec![n_rows]);

        // Track input dependencies BEFORE consuming with into_ir()
        let streams = OperationStreams::with_inputs([&logits, &targets]);

        // Consume FusionTensors → TensorIr descriptions
        let logits_ir = logits.into_ir();
        let targets_ir = targets.into_ir();

        // Create output handles (filled by FusedCEForwardOp::execute)
        let loss_id = client.create_empty_handle();
        let logsumexp_id = client.create_empty_handle();

        let loss_ir = burn_ir::TensorIr {
            id: loss_id,
            shape: out_shape.clone(),
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let lse_ir = burn_ir::TensorIr {
            id: logsumexp_id,
            shape: out_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };

        // Build custom IR — prevents fusion optimizer from merging with neighbors
        let custom_ir = CustomOpIr::new(
            "fused_ce_forward",
            &[logits_ir.clone(), targets_ir.clone()],
            &[loss_ir, lse_ir],
        );
        let ir = OperationIr::Custom(custom_ir);

        let op = FusedCEForwardOp::<R, F, I, BT> {
            logits_ir,
            targets_ir,
            loss_id,
            logsumexp_id,
            softcap,
            _phantom: PhantomData,
        };

        // register() returns Vec<FusionTensor>; outputs::<2>() extracts [loss, logsumexp]
        let [loss, logsumexp]: [_; 2] = client.register(streams, ir, op).outputs();
        (loss, logsumexp)
    }

    fn ce_backward(
        logits: FloatTensor<Self>,
        grad_loss: FloatTensor<Self>,
        logsumexp: FloatTensor<Self>,
        targets: Self::IntTensorPrimitive,
        softcap: Option<f32>,
    ) -> FloatTensor<Self> {
        use burn_cubecl::fusion::FusionCubeRuntime;
        use burn_fusion::stream::{Operation, OperationStreams};
        use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorStatus};

        // -------------------------------------------------------------------
        // Fusion Operation: backward kernel producing grad_logits
        // -------------------------------------------------------------------
        #[derive(Debug)]
        struct FusedCEBackwardOp<R, F, I, BT> {
            logits_ir: burn_ir::TensorIr,
            grad_loss_ir: burn_ir::TensorIr,
            logsumexp_ir: burn_ir::TensorIr,
            targets_ir: burn_ir::TensorIr,
            grad_logits_id: burn_ir::TensorId,
            softcap: Option<f32>,
            _phantom: PhantomData<(R, F, I, BT)>,
        }

        impl<R, F, I, BT> Operation<FusionCubeRuntime<R>> for FusedCEBackwardOp<R, F, I, BT>
        where
            R: CubeRuntime,
            F: FloatElement,
            I: IntElement,
            BT: BoolElement,
        {
            fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<R>>) {
                // Resolve all inputs to CubeTensors
                let logits: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.logits_ir);
                let grad_loss: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.grad_loss_ir);
                let logsumexp: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.logsumexp_ir);
                let targets: CubeTensor<R> =
                    handles.get_int_tensor::<CubeBackend<R, F, I, BT>>(&self.targets_ir);

                // Single fused GPU kernel: softmax + gradient + softcap chain rule
                let grad_logits =
                    fused_ce_backward(logits, grad_loss, logsumexp, targets, self.softcap);

                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.grad_logits_id,
                    grad_logits,
                );
            }
        }

        // -------------------------------------------------------------------
        // Setup: register inputs/outputs with fusion stream
        // -------------------------------------------------------------------
        let client = logits.client.clone();
        let out_dtype = logits.dtype;

        // Output shape: same as logits [rows, vocab]
        let grad_shape = logits.shape();

        // Track all 4 inputs before consuming
        let streams = OperationStreams::with_inputs([&logits, &grad_loss, &logsumexp, &targets]);

        // Consume FusionTensors → TensorIr descriptions
        let logits_ir = logits.into_ir();
        let grad_loss_ir = grad_loss.into_ir();
        let logsumexp_ir = logsumexp.into_ir();
        let targets_ir = targets.into_ir();

        // Create output handle
        let grad_logits_id = client.create_empty_handle();
        let grad_logits_ir = burn_ir::TensorIr {
            id: grad_logits_id,
            shape: grad_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };

        // Build custom IR
        let custom_ir = CustomOpIr::new(
            "fused_ce_backward",
            &[
                logits_ir.clone(),
                grad_loss_ir.clone(),
                logsumexp_ir.clone(),
                targets_ir.clone(),
            ],
            &[grad_logits_ir],
        );
        let ir = OperationIr::Custom(custom_ir);

        let op = FusedCEBackwardOp::<R, F, I, BT> {
            logits_ir,
            grad_loss_ir,
            logsumexp_ir,
            targets_ir,
            grad_logits_id,
            softcap,
            _phantom: PhantomData,
        };

        client.register(streams, ir, op).output()
    }
}
