use burn_backend::{
    Backend, ExecutionError, TensorData, TensorPrimitive, get_device_settings,
    ops::{FloatTensorOps, QTensorOps},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

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
    /// This is the critical path for LoRA training with quantized frozen weights:
    /// - Frozen weights are stored as QFloat (quantized)
    /// - Activations (and LoRA adapter outputs) are Float with gradient tracking
    /// - Forward:  dequantize(weight) @ activation  (via float_matmul with grad tracking)
    /// - Backward: grad_input = grad @ dequantize(weight)^T
    ///   (weight has no grad since it's an untracked leaf)
    ///
    /// QFloat operands are dequantized on the inner backend and wrapped as untracked
    /// leaf tensors before calling the gradient-tracked `float_matmul`.
    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        match (lhs, rhs) {
            // Both float — delegate to tracked float_matmul
            (TensorPrimitive::Float(lhs), TensorPrimitive::Float(rhs)) => {
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(lhs, rhs))
            }
            // Float @ QFloat — dequantize RHS (frozen weights), tracked matmul for LHS gradient
            (TensorPrimitive::Float(float_ad), TensorPrimitive::QFloat(q_tensor)) => {
                let device = B::q_device(&q_tensor);
                let dtype = get_device_settings::<B>(&device).float_dtype;
                let dequant_rhs = AutodiffTensor::new(B::dequantize(q_tensor, dtype));
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(
                    float_ad,
                    dequant_rhs,
                ))
            }
            // QFloat @ Float — dequantize LHS (frozen weights), tracked matmul for RHS gradient
            (TensorPrimitive::QFloat(q_tensor), TensorPrimitive::Float(float_ad)) => {
                let device = B::q_device(&q_tensor);
                let dtype = get_device_settings::<B>(&device).float_dtype;
                let dequant_lhs = AutodiffTensor::new(B::dequantize(q_tensor, dtype));
                TensorPrimitive::Float(<Self as FloatTensorOps<Self>>::float_matmul(
                    dequant_lhs,
                    float_ad,
                ))
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
