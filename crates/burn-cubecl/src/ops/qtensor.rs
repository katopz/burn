use burn_backend::{
    Bytes, DType, ExecutionError, QTensorPrimitive, Shape, Slice, TensorData, TensorMetadata,
    TensorPrimitive,
    ops::QTensorOps,
    quantization::{
        QParamTensor, QuantLevel, QuantMode, QuantParam, QuantPropagation, QuantScheme, QuantValue,
        QuantizationParametersPrimitive, params_shape,
    },
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, Metadata};
use cubecl::server::{MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutStrategy};
use cubecl::{e2m1x2, quant::scheme::QuantStore};

use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{self, matmul::MatmulStrategy},
    tensor::{CubeTensor, QParams},
};

use super::{into_data, permute, swap_dims};

/// Create a quantized tensor with packed values (u32).
fn new_qtensor_optimized<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    new_qtensor(data, shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, Some(data), kind)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor_optimized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    empty_qtensor(shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, None, kind)
}

fn new_quantized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    data: Option<Bytes>,
    alloc_kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();
    let is_affine = scheme.mode == QuantMode::Affine;

    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::PackedU32(_) => {
            if !shape_last.is_multiple_of(num_quants) {
                panic!(
                    "Can't pack {scheme:?} into u32: last dim {shape_last} not multiple of {num_quants} (shape={shape:?})"
                )
            }
            shape_value[rank - 1] = shape_last.div_ceil(num_quants);
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::E2M1 => {
                panic!("Can't store native sub-byte values")
            }
        },
        QuantStore::PackedNative(_) => match scheme.value {
            QuantValue::E2M1 => size_of::<e2m1x2>(),
            other => panic!("{other:?} doesn't support native packing"),
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
        // Represented by U8 and reinterpreted in the kernel
        QuantParam::UE8M0 | QuantParam::UE4M3 => DType::U8,
    };

    let scales_shape = params_shape(&shape, scheme.level);
    let data_desc = MemoryLayoutDescriptor::new(alloc_kind, shape_value.clone(), data_size);
    let scales_desc =
        MemoryLayoutDescriptor::new(alloc_kind, scales_shape.clone(), scales_dtype.size());

    // For affine mode, biases have the same shape/dtype as scales
    let biases_desc = is_affine.then(|| {
        MemoryLayoutDescriptor::new(alloc_kind, scales_shape.clone(), scales_dtype.size())
    });

    // Allocate tensors: [data, biases?, scales]
    // Serialization format: [values | biases | scales] (scales always last)
    let mut tensors = match (data, is_affine) {
        // Affine with data: [values | biases | scales]
        (Some(data), true) => {
            let num_bytes = shape_value.num_elements() * data_size;
            let params_bytes = scales_shape.num_elements() * scales_dtype.size();

            match data.split(num_bytes) {
                Ok((bytes_data, bytes_params)) => match bytes_params.split(params_bytes) {
                    Ok((bytes_biases, bytes_scales)) => client.create_tensors(vec![
                        (data_desc, bytes_data),
                        (biases_desc.unwrap(), bytes_biases),
                        (scales_desc, bytes_scales),
                    ]),
                    Err((params, _)) => client.create_tensors_from_slices(vec![
                        (data_desc, &bytes_data),
                        (biases_desc.unwrap(), &params[..params_bytes]),
                        (scales_desc, &params[params_bytes..]),
                    ]),
                },
                Err((data, _)) => client.create_tensors_from_slices(vec![
                    (data_desc, &data[..num_bytes]),
                    (
                        biases_desc.unwrap(),
                        &data[num_bytes..num_bytes + params_bytes],
                    ),
                    (scales_desc, &data[num_bytes + params_bytes..]),
                ]),
            }
        }
        // Symmetric with data: [values | scales]
        (Some(data), false) => {
            let num_bytes = shape_value.num_elements() * data_size;

            match data.split(num_bytes) {
                Ok((bytes_data, bytes_scales)) => client
                    .create_tensors(vec![(data_desc, bytes_data), (scales_desc, bytes_scales)]),
                Err((data, _)) => client.create_tensors_from_slices(vec![
                    (data_desc, &data[..num_bytes]),
                    (scales_desc, &data[num_bytes..]),
                ]),
            }
        }
        // Affine empty: allocate data + biases + scales
        (None, true) => client.empty_tensors(vec![data_desc, biases_desc.unwrap(), scales_desc]),
        // Symmetric empty: allocate data + scales
        (None, false) => client.empty_tensors(vec![data_desc, scales_desc]),
    };

    // Extract tensors in reverse order: scales (last), biases (middle), data (first)
    let MemoryLayout {
        memory: scales_handle,
        strides: scales_strides,
    } = tensors.remove(if is_affine { 2 } else { 1 });

    let biases_param = if is_affine {
        let MemoryLayout {
            memory: biases_handle,
            strides: biases_strides,
        } = tensors.remove(1);
        Some(QParamTensor {
            offset_start: biases_handle.offset_start.unwrap_or(0) as usize,
            offset_end: biases_handle.offset_end.unwrap_or(0) as usize,
            metadata: Metadata::new(scales_shape.clone(), biases_strides),
            dtype: scales_dtype,
        })
    } else {
        None
    };

    let MemoryLayout { memory, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        metadata: Metadata::new(scales_shape, scales_strides),
        dtype: scales_dtype,
    };
    let qparams = QParams {
        scales,
        biases: biases_param,
    };

    CubeTensor::new_quantized(
        client,
        memory,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}

impl<R, F, I, BT> QTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
                    level: QuantLevel::Tensor | QuantLevel::Block(_),
                    mode: QuantMode::Symmetric | QuantMode::Affine,
                    value:
                        QuantValue::Q8F
                        | QuantValue::Q8S
                        | QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S
                        | QuantValue::E4M3
                        | QuantValue::E5M2
                        | QuantValue::E2M1,
                    ..
                } => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor_optimized(data.bytes, data.shape.clone(), scheme, device)
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    // TODO: quantize_dynamic (we can compute min-max on the fly and scale, especially when not per-tensor)

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize(tensor, scheme, qparams.scales, qparams.biases)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        kernel::quantization::dequantize(tensor, dtype.into())
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        if tensor.qparams.is_none() {
            return into_data(tensor).await;
        }

        let (shape, dtype) = (tensor.shape(), tensor.dtype);
        let (values, scales) = tensor.quantized_handles().unwrap();

        let mut data_values = into_data(values).await?;
        let data_scales = into_data(scales).await?;

        // Serialization format: [values | biases | scales] for affine, [values | scales] for symmetric
        if let Some(biases) = tensor.biases() {
            let data_biases = into_data(biases).await?;
            data_values.bytes.extend_from_byte_slice(&data_biases.bytes);
        }
        data_values.bytes.extend_from_byte_slice(&data_scales.bytes);

        Ok(TensorData {
            bytes: data_values.bytes,
            shape,
            dtype,
        })
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        permute(tensor, axes)
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_gather(
        _dim: usize,
        _tensor: QuantizedTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(_tensor: QuantizedTensor<Self>, _slices: &[Slice]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let (propagation, scheme) = match (&lhs, &rhs) {
            (TensorPrimitive::QFloat(lhs), _) => (lhs.propagation(), *lhs.scheme()),
            (_, TensorPrimitive::QFloat(rhs)) => (rhs.propagation(), *rhs.scheme()),
            _ => unreachable!(),
        };

        // Inherit precision for mixed inputs, default to `FloatElem` for fully quantized.
        let out_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(lhs), _) => lhs.dtype,
            (_, TensorPrimitive::Float(rhs)) => rhs.dtype,
            _ => F::dtype(),
        };

        let (_lhs_dtype, lhs) = match lhs {
            TensorPrimitive::Float(lhs) => (lhs.dtype, lhs),
            TensorPrimitive::QFloat(lhs) => (out_dtype, lhs),
        };
        let (_rhs_dtype, rhs) = match rhs {
            TensorPrimitive::Float(rhs) => (rhs.dtype, rhs),
            TensorPrimitive::QFloat(rhs) => (out_dtype, rhs),
        };

        let out =
            kernel::matmul::matmul(lhs, rhs, None, MatmulStrategy::default(), out_dtype).unwrap();

        match propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
