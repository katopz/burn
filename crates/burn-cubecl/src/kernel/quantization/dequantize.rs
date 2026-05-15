use crate::tensor::CubeTensor;
use crate::{CubeRuntime, ops::numeric::empty_device_dtype};
use burn_backend::{DType, TensorMetadata};
use cubecl::quant::scheme::QuantMode;

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    let bias_binding = match scheme.mode {
        QuantMode::Affine => tensor.biases(),
        QuantMode::Symmetric => None,
    };

    cubek::quantization::dequantize::launch_ref_with_bias(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        bias_binding.map(|b| b.binding()),
        &scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
