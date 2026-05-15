use crate::CubeRuntime;
use crate::ops::numeric::empty_device_dtype;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::{TensorMetadata, quantization::QuantScheme};
use cubecl::quant::scheme::QuantMode;

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;

    // Extract bias binding before creating out_bias to avoid borrow conflicts
    let bias_binding = bias.map(|b| b.binding());

    // For affine mode, allocate a separate output buffer for biases (same shape as scales)
    let out_bias = match (&bias_binding, scheme.mode) {
        (Some(_), QuantMode::Affine) => {
            let bias_out = empty_device_dtype::<R>(
                out_params.client.clone(),
                out_params.device.clone(),
                out_params.shape().clone(),
                out_params.dtype,
            );
            Some(bias_out.binding())
        }
        _ => None,
    };

    cubek::quantization::quantize::launch_ref(
        &output.client,
        tensor.binding(),
        out_values.binding(),
        scale.binding(),
        out_params.binding(),
        scheme,
        dtype.into(),
        bias_binding,
        out_bias,
    )
    .expect("Kernel to never fail");

    output
}
