use super::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_matmul() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[11.0, 5.0], [11.0, 5.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[3.0, 3.0], [10.0, 10.0]]), false);
    tensor_3
        .to_data()
        .assert_eq(&TensorData::from([[18.0, 28.0], [14.0, 23.0]]), false);
}

#[test]
fn test_matmul_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3);

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[44.0, 20.0], [44.0, 20.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[56.0, 56.0], [16.0, 16.0]]), false);
}

#[test]
fn test_matmul_complex_2() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.matmul(tensor_3.clone());
    let tensor_6 = tensor_1.clone().matmul(tensor_5);

    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[800.0, 792.0], [360.0, 592.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[264., 264.0], [344.0, 344.0]]), false);
}

/// Test gradient flow through quantized matmul — the LoRA training pattern where
/// frozen weights are quantized and activations need gradients.
///
/// NOTE: This test requires the `cube` feature (which includes `burn-ndarray`)
/// because the Dispatch backend's autodiff path for quantized tensors needs
/// direct backend access. The core implementation in `burn-autodiff/src/ops/qtensor.rs`
/// delegates dequantize to the inner backend and wraps the result as an untracked
/// `AutodiffTensor`, then calls `float_matmul` for proper gradient tracking.
///
/// When using the Metal backend directly (as in `sft-train`), the path is:
/// `Autodiff<Metal<f16>>` → `AutodiffTensor::new(B::dequantize(...))` → `float_matmul`
/// which works without Dispatch's macro-based dispatch machinery.
#[cfg(feature = "burn-ndarray")]
#[test]
fn should_diff_matmul_with_quantized_rhs() {
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use burn_tensor::quantization::{QuantScheme, QuantStore, QuantValue};

    type B = Autodiff<NdArray<f32>>;

    let device: burn_ndarray::NdArrayDevice = Default::default();

    // Tracked float tensor (activations / LoRA output)
    let tensor_1 = Tensor::<B, 2>::from_data([[1.0, 7.0], [2.0, 3.0]], &device).require_grad();

    // Quantize frozen weights — goes through Autodiff<NdArray>::quantize_dynamic
    // which extracts the inner primitive and delegates to NdArray::quantize_dynamic
    let tensor_2_float = Tensor::<B, 2>::from_data([[4.0, 7.0], [2.0, 3.0]], &device);
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native);
    let tensor_2 = tensor_2_float.quantize_dynamic(&scheme);

    // Forward: output = lhs @ dequantize(rhs_quantized)
    // Autodiff<B>::q_matmul dequantizes RHS via B::dequantize, wraps as untracked
    // AutodiffTensor, then calls float_matmul for gradient tracking.
    // Backward: grad_lhs = grad_output @ dequantize(rhs)^T
    let tensor_3 = tensor_1.clone().matmul(tensor_2);
    let grads = tensor_3.backward();

    // Gradient must flow to tensor_1 (the tracked float operand)
    let grad_1 = tensor_1.grad(&grads).unwrap();

    let grad_data = grad_1.to_data();
    let values: Vec<f32> = grad_data.to_vec().unwrap();

    // Verify gradient shape
    assert_eq!(grad_data.shape.dims(), [2, 2]);

    // Compute reference gradient using dequantized weights on inner backend.
    // For Y = X @ W, grad_X = grad_Y @ W^T.
    // Since backward() on the output implicitly uses ones as grad_Y:
    //   grad_X = ones(2,2) @ W_dequant^T
    use burn_tensor::backend::AutodiffBackend;
    type Inner = <B as AutodiffBackend>::InnerBackend;

    // Dequantize on inner backend to get exact scale-compensated values
    let w_float_inner = Tensor::<Inner, 2>::from_data([[4.0, 7.0], [2.0, 3.0]], &device);
    let scheme_inner = QuantScheme::default()
        .with_value(QuantValue::Q8S)
        .with_store(QuantStore::Native);
    let w_q_inner = w_float_inner.quantize_dynamic(&scheme_inner);
    let w_dequant: Tensor<Inner, 2> = w_q_inner.dequantize();

    // Expected grad: ones(2,2) @ w_dequant^T
    let w_t = w_dequant.transpose();
    let ones = Tensor::<Inner, 2>::ones([2, 2], &device);
    let expected = ones.matmul(w_t);
    let expected_data = expected.into_data();
    let expected_values: Vec<f32> = expected_data.to_vec().unwrap();

    assert!(
        values.iter().all(|v: &f32| v.is_finite()),
        "gradient has non-finite values: {values:?}"
    );
    // Compare with tolerance — Q8S introduces ~0.5/127 ≈ 0.4% quantization error
    for (i, (actual, expected)) in values.iter().zip(expected_values.iter()).enumerate() {
        let diff = (actual - expected).abs();
        let tolerance = expected.abs() * 0.01 + 0.05; // 1% + small absolute tolerance
        assert!(
            diff < tolerance,
            "gradient mismatch at index {i}: actual={actual:.6}, expected={expected:.6}, diff={diff:.6}, tol={tolerance:.6}"
        );
    }
}
