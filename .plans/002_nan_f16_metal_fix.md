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
- `a1603b481` feat(lora-gemma2): add per-layer NaN diagnostic for forward pass
- `54fa46472` fix(lora-gemma2): add linear_f32 mixed-precision for all projections
- `182cedb23` fix(lora-gemma2): use linear_f32 for LM head (2304→256000 matmul)
- `7e4a3eb72` fix(lora-gemma2): cast all operands to f32 in mixed-precision linear

## Tasks

- [x] 1. Implement NaN detection wrapper — instrument forward pass to check NaN after each transformer layer ✅ `a1603b4`
- [ ] 2. Run NaN detection on f16 Metal to validate forward fix and check backward *(needs user to run with data)*
- [x] 3. Evaluate flex32 backend — ❌ NOT supported on Metal/wgpu (only CUDA+CPU). BF16 also not supported on Metal. ✅ `a1603b4`
- [ ] 4. ~~If flex32 works: benchmark~~ — SKIP, flex32 not available on Metal
- [x] 5. Implement selective f32 upcast at overflow-prone layers ✅ `7e4a3eb7` (forward pass fixed, stable loss=12.45)
- [ ] 6. Fix backward pass f16 gradient overflow — NaN appears after gradient updates despite stable forward pass
- [ ] 7. Document findings and update README with recommended dtype for Metal training

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

**Option B: Selective f32 upcast (hybrid) — IMPLEMENTED ✅**
- Only viable option for Metal (no flex32, no bf16)
- Cast ALL operands (input, weight, bias, LoRA A/B) to f32 before matmul
- Pattern: `linear_f32()` / `lora_linear_f32()` — cast weight+bias to f32, compute, cast result back
- Applied to: QKV projections, output projection, MLP gate/up/down, LM head
- Forward pass now produces stable loss (12.45) for multiple steps
- **Remaining issue**: backward pass f16 gradient overflow — NaN appears after gradient updates
- Memory: weights remain f16 in storage, f32 only transient during compute

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

## Current Status (2026-05-18)

### Forward Pass — FIXED ✅
- All linear projections (QKV, output, gate, up, down, LM head) now use f32 matmul
- Forward pass produces stable loss (12.45) across multiple batches
- `linear_f32` and `lora_linear_f32` cast ALL operands to f32 (input, weight, bias, LoRA A/B)
- Previous approach of only casting input produced garbage (loss=0.0) due to burn's dtype reinterpretation

### Backward Pass — REMAINING ISSUE ❌
- NaN appears after gradient updates despite stable forward pass
- f16 gradients overflow during backward matmul (same dot product issue as forward)
- Possible fixes:
  1. Dynamic loss scaling (scale loss up before backward, scale gradients down after)
  2. Gradient clipping at a smaller threshold (tried 0.1, still NaN)
  3. Full f32 training (stable but ~5x slower, ~2x memory)
  4. Upstream fix: cubecl/cubek f32 accumulation mode for f16 matmul backward

### Next Steps
1. Run `test-nan-per-layer --dtype f16` to validate forward fix is clean
2. Try dynamic loss scaling (e.g., `loss * 1024` → backward → `grads / 1024`)
3. If loss scaling fails, fall back to f32 training (already works: ~22s/iter, stable convergence)

## Notes

- The stash in cubek repo (quantize buffer pool fix) is independent and can be applied separately
- The test-nan-f16-vs-f32.rs binary is already committed and can be used for validation
- The test-nan-per-layer binary can pinpoint any remaining forward pass issues
- burn's matmul dispatch reinterprets rhs bytes as lhs dtype — mixed dtypes in same matmul produce garbage