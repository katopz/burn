pub use burn_std::{QPARAM_ALIGN, params_shape};
use burn_std::{QuantLevel, QuantMode, QuantScheme, Shape};

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_device_settings};

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level {
            QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            QuantLevel::Block(block_size) => {
                let shape = tensor.shape();
                let rank = shape.num_dims();
                let block_dims = block_size.to_dim_vec(rank);
                let numel = shape.num_elements();

                // Build interleaved reshape: [num_blocks_0, block_0, num_blocks_1, block_1, ...]
                // Where block_i = min(shape_i, block_size_i) — handles dim < block_size (whole dim = 1 block)
                let mut reshape = Vec::with_capacity(rank * 2);
                for (&s, &b) in shape.iter().zip(block_dims.iter()) {
                    let b = b as usize;
                    let num_blocks = s.div_ceil(b);
                    let actual_block = b.min(s);
                    reshape.push(num_blocks);
                    reshape.push(actual_block);
                }

                let reshape_numel: usize = reshape.iter().product();
                assert_eq!(
                    reshape_numel, numel,
                    "Block quantization reshape mismatch: {reshape:?} ({reshape_numel} elem) != \
                     {shape:?} ({numel} elem). Each dim must be evenly divisible by block_size \
                     or smaller than block_size."
                );

                let blocked = B::float_reshape(tensor, Shape::from(reshape));

                // Take min/max over block dimensions (odd indices) from innermost to outermost.
                // Processing in reverse order ensures indices don't shift after reduction.
                let mut blocks_min = blocked.clone();
                let mut blocks_max = blocked;

                for i in (0..rank).rev() {
                    let dim = 2 * i + 1; // block dimension (odd index)
                    blocks_min = B::float_min_dim(blocks_min, dim);
                    blocks_max = B::float_max_dim(blocks_max, dim);
                }

                // Result shape = [num_blocks_0, num_blocks_1, ...] = params_shape
                (blocks_min, blocks_max)
            }
        },
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            let bool_dtype = get_device_settings::<B>(&B::float_device(&min)).bool_dtype;
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone(), bool_dtype);
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2f32.into());

            QuantizationParametersPrimitive {
                scales: B::float_div_scalar(values_range, (b - a).into()),
                biases: None,
            }
        }
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Affine,
            ..
        } => {
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Affine: scale = (max - min) / (b - a)
            // Maps the full [min, max] range to all quantization levels [a, b]
            let scale = B::float_div_scalar(B::float_sub(max.clone(), min.clone()), (b - a).into());

            // Bias maps quantization zero-point to data zero
            // dequantize(q) = scale * q + bias
            // We want q=a → min, so bias = min - scale * a
            let bias = B::float_sub(min, B::float_mul_scalar(scale.clone(), a.into()));

            QuantizationParametersPrimitive {
                scales: scale,
                biases: Some(bias),
            }
        }
    }
}
