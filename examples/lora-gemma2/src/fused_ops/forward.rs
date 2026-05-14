//! Fused CE backend implementations for CubeBackend and Fusion<CubeBackend>.
//!
//! Two implementation paths:
//! - **CubeBackend**: Direct `#[cube]` kernel dispatch (used in tests or bare backend)
//! - **Fusion<CubeBackend>**: Fusion `Operation` dispatch (what Metal/Wgpu actually use)
//!
//! Both pass `softcap: Option<f32>` to the kernel launch functions for
//! optional inline logit softcapping (Gemma 2).

use crate::fused_ops::{FusedCEBackend, FusedLoraMLPBackend, FusedLoraMLPGrads, FusedLoraMLPState};
use crate::kernel::cross_entropy::{fused_ce_backward, fused_ce_forward};
use crate::kernel::geglu::geglu_backward;
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn_backend::ops::FloatTensorOps;
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
// Fused LoRA MLP — CubeBackend impl
// ---------------------------------------------------------------------------

/// Approximate GELU activation using tanh formula.
///
/// Matches the `geglu_backward` kernel's GELU implementation for forward/backward
/// consistency: `gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
fn gelu_approximate<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    let sqrt_2_over_pi: f64 = 0.7978845608028654;
    let coeff: f64 = 0.044715;

    let x3 = B::float_powf_scalar(x.clone(), 3.0.into());
    let inner = B::float_add(x.clone(), B::float_mul_scalar(x3, coeff.into()));
    let inner = B::float_mul_scalar(inner, sqrt_2_over_pi.into());
    let tanh_val = B::float_tanh(inner);
    let one_plus_tanh = B::float_add_scalar(tanh_val, 1.0.into());
    B::float_mul(B::float_mul_scalar(x, 0.5.into()), one_plus_tanh)
}

/// Implement fused LoRA MLP backend trait for `CubeBackend` (no Fusion wrapper).
///
/// Uses standard burn `FloatTensorOps` for the forward (matmul, add, mul, gelu)
/// and the custom `geglu_backward` kernel for the elementwise backward pass.
///
/// `FloatTensor<CubeBackend<R, F, I, BT>> = CubeTensor<R>`
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusedLoraMLPBackend
    for CubeBackend<R, F, I, BT>
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
        let (output, _state) = Self::fused_lora_mlp_forward_inner(
            x, gate_a, gate_b, gate_s, up_a, up_b, up_s, down_a, down_b, down_s, gate_w, up_w,
            down_w,
        );
        output
    }

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
    ) -> (FloatTensor<Self>, FusedLoraMLPState<Self>) {
        // e = X @ (G + Ag @ Bg * sg)
        let gate_lora = Self::float_mul_scalar(Self::float_matmul(gate_a, gate_b), gate_s.into());
        let gate_w_eff = Self::float_add(gate_w, gate_lora);
        let e = Self::float_matmul(x.clone(), gate_w_eff);

        // gate = gelu_approximate(e) — tanh approximation to match backward kernel
        let gate = gelu_approximate::<Self>(e.clone());

        // g = X @ (U + Au @ Bu * su)
        let up_lora = Self::float_mul_scalar(Self::float_matmul(up_a, up_b), up_s.into());
        let up_w_eff = Self::float_add(up_w, up_lora);
        let g = Self::float_matmul(x, up_w_eff);

        // h = gate * g
        let h = Self::float_mul(gate, g.clone());

        // out = h @ (W + Aw @ Bw * sw)
        let down_lora = Self::float_mul_scalar(Self::float_matmul(down_a, down_b), down_s.into());
        let down_w_eff = Self::float_add(down_w, down_lora);
        let out = Self::float_matmul(h, down_w_eff);

        (out, FusedLoraMLPState { e, g })
    }

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
        state: FusedLoraMLPState<Self>,
        dy: FloatTensor<Self>,
    ) -> FusedLoraMLPGrads<Self> {
        let FusedLoraMLPState { e, g } = state;

        // Precompute all transposes (each weight only transposed once)
        let x_t = Self::float_transpose(x);
        let gate_a_t = Self::float_transpose(gate_a);
        let gate_b_t = Self::float_transpose(gate_b);
        let gate_w_t = Self::float_transpose(gate_w);
        let up_a_t = Self::float_transpose(up_a);
        let up_b_t = Self::float_transpose(up_b);
        let up_w_t = Self::float_transpose(up_w);
        let down_a_t = Self::float_transpose(down_a);
        let down_b_t = Self::float_transpose(down_b);
        let down_w_t = Self::float_transpose(down_w);

        // Step 1: Down projection backward → dh
        // dh = dy @ W^T + dy @ Bw^T @ Aw^T * sw
        let dh_base = Self::float_matmul(dy.clone(), down_w_t);
        let dy_bwt = Self::float_matmul(dy.clone(), down_b_t.clone());
        let dh = Self::float_add(
            dh_base,
            Self::float_mul_scalar(Self::float_matmul(dy_bwt, down_a_t.clone()), down_s.into()),
        );

        // Step 2: GeGLU backward → (h, dg, de)
        let (h, dg, de) = geglu_backward(dh, e, g);

        // Step 3: LoRA weight gradients
        let h_t = Self::float_transpose(h);

        // Down LoRA: dAw = h^T @ (dy @ Bw^T) * sw, dBw = (Aw^T @ h^T) @ dy * sw
        let d_down_a = Self::float_mul_scalar(
            Self::float_matmul(h_t.clone(), Self::float_matmul(dy.clone(), down_b_t)),
            down_s.into(),
        );
        let d_down_b = Self::float_mul_scalar(
            Self::float_matmul(Self::float_matmul(down_a_t, h_t), dy),
            down_s.into(),
        );

        // Up LoRA: dAu = X^T @ (dg @ Bu^T) * su, dBu = (Au^T @ X^T) @ dg * su
        let d_up_a = Self::float_mul_scalar(
            Self::float_matmul(x_t.clone(), Self::float_matmul(dg.clone(), up_b_t.clone())),
            up_s.into(),
        );
        let d_up_b = Self::float_mul_scalar(
            Self::float_matmul(Self::float_matmul(up_a_t.clone(), x_t.clone()), dg.clone()),
            up_s.into(),
        );

        // Gate LoRA: dAg = X^T @ (de @ Bg^T) * sg, dBg = (Ag^T @ X^T) @ de * sg
        let d_gate_a = Self::float_mul_scalar(
            Self::float_matmul(
                x_t.clone(),
                Self::float_matmul(de.clone(), gate_b_t.clone()),
            ),
            gate_s.into(),
        );
        let d_gate_b = Self::float_mul_scalar(
            Self::float_matmul(Self::float_matmul(gate_a_t.clone(), x_t), de.clone()),
            gate_s.into(),
        );

        // Step 4: Input gradient dX
        // dX = de @ G^T + de @ Bg^T @ Ag^T * sg + dg @ U^T + dg @ Bu^T @ Au^T * su
        let dx_gate = Self::float_add(
            Self::float_matmul(de.clone(), gate_w_t),
            Self::float_mul_scalar(
                Self::float_matmul(Self::float_matmul(de, gate_b_t), gate_a_t),
                gate_s.into(),
            ),
        );
        let dx_up = Self::float_add(
            Self::float_matmul(dg.clone(), up_w_t),
            Self::float_mul_scalar(
                Self::float_matmul(Self::float_matmul(dg, up_b_t), up_a_t),
                up_s.into(),
            ),
        );
        let dx = Self::float_add(dx_gate, dx_up);

        FusedLoraMLPGrads {
            dx,
            d_gate_a,
            d_gate_b,
            d_up_a,
            d_up_b,
            d_down_a,
            d_down_b,
        }
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
        // Logsumexp stays f32 — fused_ce_forward upcasts to f32 internally
        // and returns logsumexp as f32 (only loss is cast back to original dtype).
        let lse_ir = burn_ir::TensorIr {
            id: logsumexp_id,
            shape: out_shape,
            dtype: burn::tensor::DType::F32,
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

// ---------------------------------------------------------------------------
// Fused LoRA MLP — Fusion<CubeBackend> impl
// ---------------------------------------------------------------------------

/// Implement [`FusedLoraMLPBackend`] for `Fusion<CubeBackend<...>>`.
///
/// Wraps the CubeBackend implementation in fusion [`Operation`]s so the fused
/// MLP forward and backward are dispatched as single operations in the fusion
/// stream — no intermediate tensor materialization.
///
/// # Why [`CustomOpIr`]
///
/// The fusion optimizer uses [`OperationIr`] to decide which ops to fuse.
/// Using `Custom` prevents the optimizer from splitting our fused MLP into
/// individual matmul/add/mul ops that would be fused incorrectly with neighbors.
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm",
))]
impl<R, F, I, BT> FusedLoraMLPBackend for burn_fusion::Fusion<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
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
        let (output, _state) = Self::fused_lora_mlp_forward_inner(
            x, gate_a, gate_b, gate_s, up_a, up_b, up_s, down_a, down_b, down_s, gate_w, up_w,
            down_w,
        );
        output
    }

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
    ) -> (FloatTensor<Self>, FusedLoraMLPState<Self>) {
        use burn_cubecl::fusion::FusionCubeRuntime;
        use burn_fusion::stream::{Operation, OperationStreams};
        use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorStatus};

        // ---------------------------------------------------------------
        // Fusion Operation: forward producing 3 outputs (out, e, g)
        // ---------------------------------------------------------------
        #[derive(Debug)]
        struct FusedLoraMLPForwardOp<R, F, I, BT> {
            x_ir: burn_ir::TensorIr,
            gate_a_ir: burn_ir::TensorIr,
            gate_b_ir: burn_ir::TensorIr,
            up_a_ir: burn_ir::TensorIr,
            up_b_ir: burn_ir::TensorIr,
            down_a_ir: burn_ir::TensorIr,
            down_b_ir: burn_ir::TensorIr,
            gate_w_ir: burn_ir::TensorIr,
            up_w_ir: burn_ir::TensorIr,
            down_w_ir: burn_ir::TensorIr,
            out_id: burn_ir::TensorId,
            e_id: burn_ir::TensorId,
            g_id: burn_ir::TensorId,
            gate_s: f64,
            up_s: f64,
            down_s: f64,
            _phantom: PhantomData<(R, F, I, BT)>,
        }

        impl<R, F, I, BT> Operation<FusionCubeRuntime<R>> for FusedLoraMLPForwardOp<R, F, I, BT>
        where
            R: CubeRuntime,
            F: FloatElement,
            I: IntElement,
            BT: BoolElement,
        {
            fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<R>>) {
                let x: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.x_ir);
                let gate_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_a_ir);
                let gate_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_b_ir);
                let up_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_a_ir);
                let up_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_b_ir);
                let down_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_a_ir);
                let down_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_b_ir);
                let gate_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_w_ir);
                let up_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_w_ir);
                let down_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_w_ir);

                let (out, state) = CubeBackend::<R, F, I, BT>::fused_lora_mlp_forward_inner(
                    x,
                    gate_a,
                    gate_b,
                    self.gate_s,
                    up_a,
                    up_b,
                    self.up_s,
                    down_a,
                    down_b,
                    self.down_s,
                    gate_w,
                    up_w,
                    down_w,
                );

                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(&self.out_id, out);
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(&self.e_id, state.e);
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(&self.g_id, state.g);
            }
        }

        // ---------------------------------------------------------------
        // Setup: register inputs/outputs with fusion stream
        // ---------------------------------------------------------------
        let client = x.client.clone();
        let out_dtype = x.dtype;

        // Output shapes from input dimensions
        let [n, _d_in] = x.shape().dims::<2>();
        let [_d_in, d_mid] = gate_w.shape().dims::<2>();
        let [_d_mid, d_out] = down_w.shape().dims::<2>();

        let out_shape = Shape::new([n, d_out]);
        let e_shape = Shape::new([n, d_mid]);
        let g_shape = Shape::new([n, d_mid]);

        // Track input dependencies BEFORE consuming with into_ir()
        let streams = OperationStreams::with_inputs([
            &x, &gate_a, &gate_b, &up_a, &up_b, &down_a, &down_b, &gate_w, &up_w, &down_w,
        ]);

        // Consume FusionTensors → TensorIr descriptions
        let x_ir = x.into_ir();
        let gate_a_ir = gate_a.into_ir();
        let gate_b_ir = gate_b.into_ir();
        let up_a_ir = up_a.into_ir();
        let up_b_ir = up_b.into_ir();
        let down_a_ir = down_a.into_ir();
        let down_b_ir = down_b.into_ir();
        let gate_w_ir = gate_w.into_ir();
        let up_w_ir = up_w.into_ir();
        let down_w_ir = down_w.into_ir();

        // Create output handles (filled by FusedLoraMLPForwardOp::execute)
        let out_id = client.create_empty_handle();
        let e_id = client.create_empty_handle();
        let g_id = client.create_empty_handle();

        let out_ir = burn_ir::TensorIr {
            id: out_id,
            shape: out_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let e_ir = burn_ir::TensorIr {
            id: e_id,
            shape: e_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let g_ir = burn_ir::TensorIr {
            id: g_id,
            shape: g_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };

        // Build custom IR — prevents fusion optimizer from merging with neighbors
        let custom_ir = CustomOpIr::new(
            "fused_lora_mlp_forward",
            &[
                x_ir.clone(),
                gate_a_ir.clone(),
                gate_b_ir.clone(),
                up_a_ir.clone(),
                up_b_ir.clone(),
                down_a_ir.clone(),
                down_b_ir.clone(),
                gate_w_ir.clone(),
                up_w_ir.clone(),
                down_w_ir.clone(),
            ],
            &[out_ir, e_ir, g_ir],
        );
        let ir = OperationIr::Custom(custom_ir);

        let op = FusedLoraMLPForwardOp::<R, F, I, BT> {
            x_ir,
            gate_a_ir,
            gate_b_ir,
            up_a_ir,
            up_b_ir,
            down_a_ir,
            down_b_ir,
            gate_w_ir,
            up_w_ir,
            down_w_ir,
            out_id,
            e_id,
            g_id,
            gate_s,
            up_s,
            down_s,
            _phantom: PhantomData,
        };

        let [out, e, g]: [_; 3] = client.register(streams, ir, op).outputs();
        (out, FusedLoraMLPState { e, g })
    }

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
        state: FusedLoraMLPState<Self>,
        dy: FloatTensor<Self>,
    ) -> FusedLoraMLPGrads<Self> {
        use burn_cubecl::fusion::FusionCubeRuntime;
        use burn_fusion::stream::{Operation, OperationStreams};
        use burn_ir::{CustomOpIr, OperationIr, OperationOutput, TensorStatus};

        let FusedLoraMLPState { e, g } = state;

        // ---------------------------------------------------------------
        // Fusion Operation: backward producing 7 gradient outputs
        // ---------------------------------------------------------------
        #[derive(Debug)]
        struct FusedLoraMLPBackwardOp<R, F, I, BT> {
            // 13 input IRs
            x_ir: burn_ir::TensorIr,
            gate_a_ir: burn_ir::TensorIr,
            gate_b_ir: burn_ir::TensorIr,
            up_a_ir: burn_ir::TensorIr,
            up_b_ir: burn_ir::TensorIr,
            down_a_ir: burn_ir::TensorIr,
            down_b_ir: burn_ir::TensorIr,
            gate_w_ir: burn_ir::TensorIr,
            up_w_ir: burn_ir::TensorIr,
            down_w_ir: burn_ir::TensorIr,
            e_ir: burn_ir::TensorIr,
            g_ir: burn_ir::TensorIr,
            dy_ir: burn_ir::TensorIr,
            // 7 output IDs
            dx_id: burn_ir::TensorId,
            d_gate_a_id: burn_ir::TensorId,
            d_gate_b_id: burn_ir::TensorId,
            d_up_a_id: burn_ir::TensorId,
            d_up_b_id: burn_ir::TensorId,
            d_down_a_id: burn_ir::TensorId,
            d_down_b_id: burn_ir::TensorId,
            // 3 scalars
            gate_s: f64,
            up_s: f64,
            down_s: f64,
            _phantom: PhantomData<(R, F, I, BT)>,
        }

        impl<R, F, I, BT> Operation<FusionCubeRuntime<R>> for FusedLoraMLPBackwardOp<R, F, I, BT>
        where
            R: CubeRuntime,
            F: FloatElement,
            I: IntElement,
            BT: BoolElement,
        {
            fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<R>>) {
                let x: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.x_ir);
                let gate_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_a_ir);
                let gate_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_b_ir);
                let up_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_a_ir);
                let up_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_b_ir);
                let down_a: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_a_ir);
                let down_b: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_b_ir);
                let gate_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.gate_w_ir);
                let up_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.up_w_ir);
                let down_w: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.down_w_ir);
                let e: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.e_ir);
                let g: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.g_ir);
                let dy: CubeTensor<R> =
                    handles.get_float_tensor::<CubeBackend<R, F, I, BT>>(&self.dy_ir);

                let grads = CubeBackend::<R, F, I, BT>::fused_lora_mlp_backward(
                    x,
                    gate_a,
                    gate_b,
                    self.gate_s,
                    up_a,
                    up_b,
                    self.up_s,
                    down_a,
                    down_b,
                    self.down_s,
                    gate_w,
                    up_w,
                    down_w,
                    FusedLoraMLPState { e, g },
                    dy,
                );

                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(&self.dx_id, grads.dx);
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_gate_a_id,
                    grads.d_gate_a,
                );
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_gate_b_id,
                    grads.d_gate_b,
                );
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_up_a_id,
                    grads.d_up_a,
                );
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_up_b_id,
                    grads.d_up_b,
                );
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_down_a_id,
                    grads.d_down_a,
                );
                handles.register_float_tensor::<CubeBackend<R, F, I, BT>>(
                    &self.d_down_b_id,
                    grads.d_down_b,
                );
            }
        }

        // ---------------------------------------------------------------
        // Setup: register inputs/outputs with fusion stream
        // ---------------------------------------------------------------
        let client = x.client.clone();
        let out_dtype = x.dtype;

        // Gradient shapes match the corresponding weight shapes
        let dx_shape = x.shape();
        let d_gate_a_shape = gate_a.shape();
        let d_gate_b_shape = gate_b.shape();
        let d_up_a_shape = up_a.shape();
        let d_up_b_shape = up_b.shape();
        let d_down_a_shape = down_a.shape();
        let d_down_b_shape = down_b.shape();

        // Track all 13 inputs before consuming
        let streams = OperationStreams::with_inputs([
            &x, &gate_a, &gate_b, &up_a, &up_b, &down_a, &down_b, &gate_w, &up_w, &down_w, &e, &g,
            &dy,
        ]);

        // Consume FusionTensors → TensorIr descriptions
        let x_ir = x.into_ir();
        let gate_a_ir = gate_a.into_ir();
        let gate_b_ir = gate_b.into_ir();
        let up_a_ir = up_a.into_ir();
        let up_b_ir = up_b.into_ir();
        let down_a_ir = down_a.into_ir();
        let down_b_ir = down_b.into_ir();
        let gate_w_ir = gate_w.into_ir();
        let up_w_ir = up_w.into_ir();
        let down_w_ir = down_w.into_ir();
        let e_ir = e.into_ir();
        let g_ir = g.into_ir();
        let dy_ir = dy.into_ir();

        // Create output handles
        let dx_id = client.create_empty_handle();
        let d_gate_a_id = client.create_empty_handle();
        let d_gate_b_id = client.create_empty_handle();
        let d_up_a_id = client.create_empty_handle();
        let d_up_b_id = client.create_empty_handle();
        let d_down_a_id = client.create_empty_handle();
        let d_down_b_id = client.create_empty_handle();

        let dx_ir = burn_ir::TensorIr {
            id: dx_id,
            shape: dx_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_gate_a_ir = burn_ir::TensorIr {
            id: d_gate_a_id,
            shape: d_gate_a_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_gate_b_ir = burn_ir::TensorIr {
            id: d_gate_b_id,
            shape: d_gate_b_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_up_a_ir = burn_ir::TensorIr {
            id: d_up_a_id,
            shape: d_up_a_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_up_b_ir = burn_ir::TensorIr {
            id: d_up_b_id,
            shape: d_up_b_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_down_a_ir = burn_ir::TensorIr {
            id: d_down_a_id,
            shape: d_down_a_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };
        let d_down_b_ir = burn_ir::TensorIr {
            id: d_down_b_id,
            shape: d_down_b_shape,
            dtype: out_dtype,
            status: TensorStatus::NotInit,
        };

        // Build custom IR
        let custom_ir = CustomOpIr::new(
            "fused_lora_mlp_backward",
            &[
                x_ir.clone(),
                gate_a_ir.clone(),
                gate_b_ir.clone(),
                up_a_ir.clone(),
                up_b_ir.clone(),
                down_a_ir.clone(),
                down_b_ir.clone(),
                gate_w_ir.clone(),
                up_w_ir.clone(),
                down_w_ir.clone(),
                e_ir.clone(),
                g_ir.clone(),
                dy_ir.clone(),
            ],
            &[
                dx_ir,
                d_gate_a_ir,
                d_gate_b_ir,
                d_up_a_ir,
                d_up_b_ir,
                d_down_a_ir,
                d_down_b_ir,
            ],
        );
        let ir = OperationIr::Custom(custom_ir);

        let op = FusedLoraMLPBackwardOp::<R, F, I, BT> {
            x_ir,
            gate_a_ir,
            gate_b_ir,
            up_a_ir,
            up_b_ir,
            down_a_ir,
            down_b_ir,
            gate_w_ir,
            up_w_ir,
            down_w_ir,
            e_ir,
            g_ir,
            dy_ir,
            dx_id,
            d_gate_a_id,
            d_gate_b_id,
            d_up_a_id,
            d_up_b_id,
            d_down_a_id,
            d_down_b_id,
            gate_s,
            up_s,
            down_s,
            _phantom: PhantomData,
        };

        let [dx, d_gate_a, d_gate_b, d_up_a, d_up_b, d_down_a, d_down_b]: [_; 7] =
            client.register(streams, ir, op).outputs();

        FusedLoraMLPGrads {
            dx,
            d_gate_a,
            d_gate_b,
            d_up_a,
            d_up_b,
            d_down_a,
            d_down_b,
        }
    }
}
