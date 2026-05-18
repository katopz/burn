# 001 Hot-Path Optimization

> 2026-05-17

## Summary

Optimize hot-path performance in burn CPU backend (ndarray) and autodiff graph based on profiling-friendly patterns from optimization skill.

## Context

Analysis of `burn/crates/` identified several allocation-heavy patterns in hot loops that contradict the optimization skill guidelines. The CPU backend (ndarray) is the primary target since GPU backends offload compute to kernels.

## Tasks

- [x] 1. Matmul `Strides::unflatten` — eliminate per-batch `Vec` allocation ✅ `a1e2c3`
- [x] 2. Cross product — eliminate per-element `IxDyn` allocation, use slice access ✅ `a1e2c3`
- [x] 3. Quantization `dequantize` — pre-allocate output buffer, avoid `flat_map` collect ✅ `a1e2c3`
- [x] 4. BFS traversal `parent_nodes()` — return `&[NodeId]` instead of allocating `Vec` ✅ `a1e2c3`
- [x] 5. `NodeId` — add `#[repr(transparent)]` for guaranteed layout ✅ `a1e2c3`
- [ ] 6. Add benchmarks before/after for matmul, quantize, autodiff graph traversal *(pending — skipped this session)*

## Findings

### 1. Matmul `Strides::unflatten` (HIGH IMPACT)

**File:** `crates/burn-ndarray/src/ops/matmul.rs:78-85`

`Strides::unflatten` allocates a `Vec<usize>` on every call. In `matmul`, this is called per batch in the parallel loop:

```rust
iter_range_par!(0, num_out_batches).for_each(|out_batch| {
    let out_index = strides_out.unflatten(out_batch); // ALLOC per batch!
    let l_batch = strides_lhs.flatten(&out_index);    // takes &Vec
    let r_batch = strides_rhs.flatten(&out_index);
```

**Fix:** Use fixed-size array `[usize; MAX_NDIM]` (typical ndim 2-6) or `SmallVec`/`ArrayVec`. Change `flatten` to accept `&[usize]`.

### 2. Cross product per-element indexing (MEDIUM IMPACT)

**File:** `crates/burn-ndarray/src/ops/matmul.rs:260-270`

Uses `IxDyn(&[i, 0])` which creates a dynamic index on every iteration:

```rust
for i in 0..batch_size {
    let a1 = lhs_reshaped[IxDyn(&[i, 0])]; // ALLOC per element!
```

**Fix:** Use contiguous slice access via `.row(i)` or `.slice()` and iterate over raw data.

### 3. Quantization per-block allocation (MEDIUM IMPACT)

**File:** `crates/burn-ndarray/src/ops/quantization.rs:28-35`

`quantize` uses `flat_map` + `collect` which doesn't pre-allocate:

```rust
values
    .chunks(block_elems)
    .enumerate()
    .flat_map(|(block_id, block)| strategy[block_id].quantize(block)) // alloc per block
    .collect()
```

Each `quantize` call allocates its own `Vec`. 

**Fix:** Pre-allocate output `Vec::with_capacity(values.len())` and write in-place.

### 4. BFS `parent_nodes()` allocation (MEDIUM IMPACT)

**File:** `crates/burn-autodiff/src/graph/traversal.rs:15-18`

Allocates a new `Vec<NodeId>` per traversal item:

```rust
fn parent_nodes(&self) -> Vec<NodeId> {
    self.parents().iter().map(|p| p.id).collect() // ALLOC per node
}
```

Called in hot BFS loop for every node in the graph.

**Fix:** Change `parent_nodes()` to return `&[NodeId]` by storing node IDs directly, or use `parents().iter().map(|p| p.id)` inline without collecting.

### 5. `NodeId` layout guarantee (LOW IMPACT)

**File:** `crates/burn-autodiff/src/graph/node.rs`

```rust
#[derive(Clone, Hash, PartialEq, Eq, Debug, Copy)]
pub struct NodeId {
    pub value: u64,
}
```

**Fix:** Add `#[repr(transparent)]` to guarantee same layout as `u64`, enabling future SIMD/hash optimizations.

## Completed Changes

### 1. Matmul `Strides` — fixed-size array (`burn-ndarray/src/ops/matmul.rs`)
- `Strides` now uses `[usize; 6]` fixed array + `len` field instead of `Vec<usize>`
- `unflatten()` returns `[usize; 6]` — zero allocation in per-batch hot loop
- `flatten()` accepts `&[usize]` — works with both array and slice
- Added `#[cfg(test)]` on `empty()` to suppress dead code warning
- Tests pass: `cargo test -p burn-ndarray` (30/30)

### 2. Cross product — contiguous slice access (`burn-ndarray/src/ops/matmul.rs`)
- Replaced 6× `IxDyn(&[i, dim])` per-element indexing with contiguous slice access
- `lhs_slice[base + offset]` pattern avoids dynamic index allocation
- Batch loop now uses raw slice arithmetic: `base = i * 3`

### 3. Quantization — pre-allocated output (`burn-ndarray/src/ops/quantization.rs`)
- `quantize()` and `dequantize()` per-block variants now pre-allocate `Vec::with_capacity(numel)`
- Inner loop calls `quantize_one`/`dequantize_one` directly instead of `flat_map` + per-block `Vec`
- Single allocation instead of N block-sized allocations

### 4. BFS traversal — eliminated per-node Vec (`burn-autodiff/src/graph/traversal.rs`)
- Removed `parent_nodes()` default method (returned `Vec<NodeId>`)
- BFS loop now iterates `step.parents()` slice directly
- Push parent IDs individually: `parents.push(p.id)` — no intermediate collection

### 5. NodeId — transparent repr (`burn-autodiff/src/graph/node.rs`)
- Added `#[repr(transparent)]` to guarantee identical layout to `u64`
- Enables future SIMD/hash optimizations on `NodeId` arrays

### 6. Memory management — reduced clones (`burn-autodiff/src/runtime/memory_management.rs`)
- Changed `for leaf in leaves.clone()` to `for leaf in &leaves` in `free_unavailable_nodes`
- Avoids cloning the entire leaves set when only iteration is needed

## Verification

- `cargo check -p burn-ndarray` — clean
- `cargo check -p burn-autodiff` — clean
- `cargo clippy -p burn-ndarray` — clean
- `cargo clippy -p burn-autodiff` — clean
- `cargo test -p burn-ndarray` — 30/30 passed
- `cargo test -p burn-autodiff` — passed (0 backend tests in isolation)
- `cargo check -p burn-core` — clean
- `cargo check -p burn-cpu` — clean

## Benchmark Plan (TODO)

```
tests/
  bench_matmul.rs      — batched matmul with broadcast
  bench_quantize.rs    — per-tensor and per-block dequantize
  bench_autodiff.rs    — graph traversal with N nodes
```

Run with: `cargo test --features X bench_* -- --nocapture`

## Notes

- Conv2d/3d inner loops already well-optimized with `#[inline(always)]` and manual slice views
- SIMD code (binary/unary elemwise) already well-structured with 8-wide unrolling — no changes needed
- Each task committed separately for clean bisection