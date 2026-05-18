# 002 NaN f16 Metal Fix

> 2026-05-18

## Summary

Fix f16 NaN overflow in Gemma 2 2B LoRA training on Metal (wgpu) backend by diagnosing the exact layer where overflow originates and evaluating flex32 as the solution.

## Context

f16 Gemma 2 2B training on Metal produces NaN from step 1 in the **forward pass**. This is NOT a backward-pass or gradient issue. The model's hidden_size=2304 means large dot products that overflow f16's limited range (max ±65504). Gemma 2's logit softcapping (at 30.0) is insufficient when intermediate activations overflow before reaching the capping stage.

### What We Know
- NaN appears in forward pass, not backward
- Even with aggressive gradient clipping (0.1), low LR (1e-5), short sequences (256), low LoRA rank (4) — still NaN
- cubek `Acc = (f16, f32)` mixed-precision accumulation made things worse (loss → 0.0)
- f32 training works perfectly (~22s/iter, stable loss)
- f16 training is ~5x faster (~3-4s/iter) but produces NaN
- f32 upcast before log_softmax prevents NaN in the CE loss computation (committed)
- The NaN originates BEFORE the loss computation — in the model's intermediate layers

### Related Commits
- `92ecf2c36` feat(train): log training/validation metrics in CLI renderer
- `a1d8dc6de` fix(lora-gemma2): f32 upcast before log_softmax to prevent f16 overflow
- `267bf97bf` refactor(lora-gemma2): remove BalancedCheckpointing, clarify f16 NaN test config
- `71fe61d05` test(lora-gemma2): add NaN diagnostic test comparing f16 vs f32 Metal

## Tasks

- [x] 1. Implement NaN detection wrapper — instrument forward pass to check NaN after each transformer layer ✅ `a1603b4`
- [ ] 2. Run NaN detection on f16 Metal to identify exact overflow layer and operation *(needs user to run with data)*
- [x] 3. Evaluate flex32 backend — ❌ NOT supported on Metal/wgpu (only CUDA+CPU). BF16 also not supported on Metal. ✅ `a1603b4`
- [ ] 4. ~~If flex32 works: benchmark~~ — SKIP, flex32 not available on Metal
- [ ] 5. Implement selective f32 upcast at overflow-prone layers (hybrid approach — only viable option for Metal)
- [ ] 6. Document findings and update README with recommended dtype for Metal training

## Technical Approach

### Phase 1: NaN Detection (Tasks 1-2)

Create a diagnostic wrapper that runs the model forward pass and checks for NaN/inf after each sub-layer:

```
Input Embeddings → [check NaN]
  └─ Layer 0
       ├─ RMSNorm → [check NaN]
       ├─ Attention (QKV proj) → [check NaN]
       ├─ Softmax → [check NaN]
       ├─ Attention output proj → [check NaN]
       ├─ RMSNorm (post-attn) → [check NaN]
       ├─ MLP (gate/up proj) → [check NaN]
       ├─ Activation (gelu) → [check NaN]
       └─ MLP (down proj) → [check NaN]
  └─ Layer 1..25 (same checks)
  └─ Final RMSNorm → [check NaN]
  └─ LM Head → [check NaN]
```

Implementation: Add a `forward_diagnostic()` method to `Gemma2Model` that:
1. Runs each layer individually (not fused)
2. After each sub-operation, calls `contains_nan()` and `min()/max()` on the output tensor
3. Logs the first layer + operation where NaN/inf appears
4. Also logs the value range (min/max) at each stage to detect approaching overflow

This will be a separate binary `test-nan-per-layer.rs` that can run with both f16 and f32 for comparison.

### Phase 2: Fix Evaluation (Tasks 3-5)

**~~Option A: flex32~~ — NOT AVAILABLE ON METAL**
- burn's `flex32` is a 16-bit format with f32 accumulation
- Supported on CUDA and CPU only — `supports_dtype(DType::Flex32)` returns `false` for Metal
- Verified in `burn-wgpu/src/lib.rs` tests (line 141)
- ❌ Cannot use this path

**Option B: Selective f32 upcast (hybrid) — CHOSEN PATH**
- Only viable option for Metal (no flex32, no bf16)
- Cast to f32 only for operations that overflow (identified by running test-nan-per-layer)
- Pattern: `tensor.cast(F32) → op → result.cast(F16)`
- Most likely candidates: QKV projection, MLP gate/up projection (hidden_size=2304 → large dot products)
- Trade-off: slightly slower than pure f16 but much more stable
- Memory: still f16 for weights/storage, f32 only transient during compute

**~~Option C: BF16~~ — NOT AVAILABLE ON METAL**
- BF16 has same range as f32 (8 exponent bits) but lower precision (7 mantissa bits)
- `supports_dtype(DType::BF16)` returns `false` for Metal
- ❌ Cannot use this path

### Phase 3: Validation (Task 6)

Run full training for 50+ steps with the fix:
- Loss should decrease smoothly (no NaN, no inf)
- Compare convergence speed: f16-fixed vs f32 baseline
- Compare memory usage: f16-fixed vs f32 baseline
- Compare throughput: f16-fixed vs f32 baseline

## Files to Modify

- `examples/lora-gemma2/src/bin/test-nan-per-layer.rs` — ✅ DONE: per-layer NaN diagnostic binary
- `examples/lora-gemma2/src/model.rs` — ✅ DONE: `forward_diagnostic()` method + `LayerCheck` struct
- `examples/lora-gemma2/src/model.rs` — NEXT: Add selective f32 upcast to overflow-prone ops
- `examples/lora-gemma2/src/bin/sft-train.rs` — Update Backend type based on fix
- `examples/lora-gemma2/src/model_lora.rs` — Any needed upcast changes

## Success Criteria

1. f16 (or flex32) Gemma 2 2B training on Metal runs 50+ steps without NaN
2. Loss decreases smoothly (within 10% of f32 baseline convergence)
3. Throughput is at least 2x faster than f32 baseline
4. Peak memory is at least 30% less than f32 baseline

## dtype Support Matrix (Metal/wgpu)

| dtype  | Metal | CUDA | CPU | Notes                          |
|--------|-------|------|-----|--------------------------------|
| f16    | ✅    | ✅   | ✅  | Overflows with large dot products |
| f32    | ✅    | ✅   | ✅  | Stable but 2x memory, ~5x slower |
| flex32 | ❌    | ✅   | ✅  | 16-bit storage + f32 compute     |
| bf16   | ❌    | ✅   | ✅  | f32 range + 16-bit precision     |

Source: `burn-wgpu/src/lib.rs` `should_support_dtypes` test

## Notes

- The stash in cubek repo (quantize buffer pool fix) is independent and can be applied separately
- The test-nan-f16-vs-f32.rs binary is already committed and can be used for validation
- If the root cause is in burn's wgpu/Metal kernel implementations (not the model), this may require upstream fixes to burn or cubek
- **Next step**: User must run `test-nan-per-layer --dtype f16` to identify exact overflow layer, then we implement selective f32 upcast at that layer