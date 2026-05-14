//! Custom fused GPU kernels for Gemma 2 training speed optimization.
//!
//! Each kernel fuses multiple tensor operations into a single GPU dispatch,
//! eliminating intermediate tensor materialization and reducing kernel launch
//! overhead. This matches what unsloth does with Triton kernels.
//!
//! # Kernels
//!
//! - **cross_entropy**: Fused CE loss + Gemma 2 logit softcapping (forward + backward)
//!
//! # Reference
//!
//! - `unsloth/kernels/cross_entropy_loss.py` — Triton kernels we adapted from
//! - `burn/crates/burn-cubecl/src/kernel/ctc.rs` — shared memory + `sync_cube` patterns
//! - `burn/examples/custom-cubecl-kernel/` — `#[cube]` + autodiff backward example

pub mod cross_entropy;
pub mod geglu;

pub use cross_entropy::{fused_ce_backward, fused_ce_forward};
pub use geglu::geglu_backward;
