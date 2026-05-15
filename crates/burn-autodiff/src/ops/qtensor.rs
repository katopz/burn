use burn_backend::{
    Backend, ExecutionError, TensorData, TensorPrimitive, get_device_settings,
    ops::{FloatTensorOps, QTensorOps},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
    tensor::AutodiffTensor,
};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    /// Creates a quantized tensor from raw data — delegates to inner backend.
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_from_data(data, device)
    }

    /// Quantize a float tensor with pre-computed parameters.
    ///
    /// The autodiff tensor's inner primitive is extracted; gradient information is lost
    /// (quantization-aware training is not yet supported).
    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: burn_backend::tensor::quantization::QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let inner_qparams = burn_backend::tensor::quantization::QuantizationParametersPrimitive {
            scales: qparams.scales.primitive,
        };
        B::quantize(tensor.primitive, scheme, inner_qparams)
    }

    /// Dynamically quantize a float tensor (computes min/max range on the fly).
    ///
    /// The autodiff tensor's inner primitive is extracted; gradient information is lost.
    /// Delegates the entire computation to the inner backend to avoid dispatch mismatches
    /// when intermediate tensors (scales, offsets) are created on different backend paths.
    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        B::quantize_dynamic(tensor.primitive, scheme)
    }

    /// Dequantize a quantized tensor to float, wrapped as an untracked leaf tensor.
    ///
    /// Returns a new `AutodiffTensor` with no gradient requirement, suitable for use
    /// as a frozen weight in matmul with gradient-tracked activations.
    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        AutodiffTensor::new(B::dequantize(tensor, dtype))
    }

    /// Quantized matmul with gradient tracking for float operands.
    ///
    /// Critical optimization for LoRA training with quantized frozen weights:
    /// - Forward: delegates to inner `B::q_matmul` for fused quantized matmul
    ///   (no dequantize during forward — weights stay compressed in GPU memory)
    /// - Backward: dequantizes weights only for gradient computation
    ///   (one dequantize per backward pass, much cheaper than keeping full f16 weights)
    ///
    /// Pattern:
    ///   Forward:  output = q_matmul(float_activation, quantized_weight)  — fused kernel
    ///   Backward: grad_act = grad_output @ dequantize(weight)^T          — weight frozen, no grad
    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        match (lhs, rhs) {
            // Both float — delegate to tracked float_matmul
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(lhs, rhs))
            }

            // Float @ QFloat — fused forward via inner backend, backward dequantizes only for grad
            //
            // Memory benefit: weights stay compressed (e.g., Q8S = 2.5GB vs f16 = 5GB) during forward.
            // The dequantize only happens once during backward for the transpose matmul.
            (TensorPrimitive::Float(float_ad), TensorPrimitive::QFloat(q_tensor)) => {
                #[derive(Debug)]
                struct QMatmulFloatLhs;

                impl<B: Backend> Backward<B, 1> for QMatmulFloatLhs {
                    type State = B::QuantizedTensorPrimitive;

                    fn backward(
                        self,
                        ops: Ops<Self::State, 1>,
                        grads: &mut Gradients,
                        _checkpointer: &mut Checkpointer,
                    ) {
                        let q_rhs = ops.state;
                        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                            // grad_lhs = grad_output @ dequantize(q_weight)^T
                            let device = B::q_device(&q_rhs);
                            let dtype = get_device_settings::<B>(&device).float_dtype;
                            let dequant_rhs = B::dequantize(q_rhs, dtype);
                            let rhs_t = B::float_transpose(dequant_rhs);
                            B::float_matmul(grad, rhs_t)
                        })
                    }
                }

                match QMatmulFloatLhs
                    .prepare::<C>([float_ad.node.clone()])
                    .compute_bound()
                    .stateful()
                {
                    OpsKind::Tracked(prep) => {
                        // Clone q_tensor: one copy for forward (consumed by q_matmul),
                        // original kept for backward state.
                        let output = match B::q_matmul(
                            TensorPrimitive::Float(float_ad.primitive),
                            TensorPrimitive::QFloat(q_tensor.clone()),
                        ) {
                            TensorPrimitive::Float(f) => f,
                            TensorPrimitive::QFloat(_) => panic!(
                                "q_matmul Float@QFloat must return Float with Inhibit propagation"
                            ),
                        };
                        TensorPrimitive::Float(prep.finish(q_tensor, output))
                    }
                    OpsKind::UnTracked(prep) => {
                        let output = match B::q_matmul(
                            TensorPrimitive::Float(float_ad.primitive),
                            TensorPrimitive::QFloat(q_tensor),
                        ) {
                            TensorPrimitive::Float(f) => f,
                            TensorPrimitive::QFloat(_) => panic!(
                                "q_matmul Float@QFloat must return Float with Inhibit propagation"
                            ),
                        };
                        TensorPrimitive::Float(prep.finish(output))
                    }
                }
            }

            // QFloat @ Float — fused forward via inner backend, backward dequantizes only for grad
            //
            // Symmetric to the Float @ QFloat case above.
            (TensorPrimitive::QFloat(q_tensor), TensorPrimitive::Float(float_ad)) => {
                #[derive(Debug)]
                struct QMatmulFloatRhs;

                impl<B: Backend> Backward<B, 1> for QMatmulFloatRhs {
                    type State = B::QuantizedTensorPrimitive;

                    fn backward(
                        self,
                        ops: Ops<Self::State, 1>,
                        grads: &mut Gradients,
                        _checkpointer: &mut Checkpointer,
                    ) {
                        let q_lhs = ops.state;
                        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                            // grad_rhs = dequantize(q_weight)^T @ grad_output
                            let device = B::q_device(&q_lhs);
                            let dtype = get_device_settings::<B>(&device).float_dtype;
                            let dequant_lhs = B::dequantize(q_lhs, dtype);
                            let lhs_t = B::float_transpose(dequant_lhs);
                            B::float_matmul(lhs_t, grad)
                        })
                    }
                }

                match QMatmulFloatRhs
                    .prepare::<C>([float_ad.node.clone()])
                    .compute_bound()
                    .stateful()
                {
                    OpsKind::Tracked(prep) => {
                        let output = match B::q_matmul(
                            TensorPrimitive::QFloat(q_tensor.clone()),
                            TensorPrimitive::Float(float_ad.primitive),
                        ) {
                            TensorPrimitive::Float(f) => f,
                            TensorPrimitive::QFloat(_) => panic!(
                                "q_matmul QFloat@Float must return Float with Inhibit propagation"
                            ),
                        };
                        TensorPrimitive::Float(prep.finish(q_tensor, output))
                    }
                    OpsKind::UnTracked(prep) => {
                        let output = match B::q_matmul(
                            TensorPrimitive::QFloat(q_tensor),
                            TensorPrimitive::Float(float_ad.primitive),
                        ) {
                            TensorPrimitive::Float(f) => f,
                            TensorPrimitive::QFloat(_) => panic!(
                                "q_matmul QFloat@Float must return Float with Inhibit propagation"
                            ),
                        };
                        TensorPrimitive::Float(prep.finish(output))
                    }
                }
            }

            // Both quantized — delegate to inner backend, wrap result as untracked leaf
            (TensorPrimitive::QFloat(lhs_q), TensorPrimitive::QFloat(rhs_q)) => {
                match B::q_matmul(
                    TensorPrimitive::QFloat(lhs_q),
                    TensorPrimitive::QFloat(rhs_q),
                ) {
                    TensorPrimitive::Float(f) => TensorPrimitive::Float(AutodiffTensor::new(f)),
                    TensorPrimitive::QFloat(q) => TensorPrimitive::QFloat(q),
                }
            }
        }
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        B::q_device(tensor)
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        B::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        B::q_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        B::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[burn_std::Slice]) -> QuantizedTensor<Self> {
        B::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor, dim, out_dtype)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_expand(tensor, shape)
    }
}
