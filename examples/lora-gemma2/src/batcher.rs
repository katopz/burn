//! Supervised Fine-Tuning (SFT) batcher for LoRA training.
//!
//! Converts [`ChatItem`] conversations into tensor batches for next-token
//! prediction training:
//!
//! 1. Format each conversation with the Gemma chat template
//! 2. Tokenize the formatted text
//! 3. Truncate to `max_seq_length` tokens
//! 4. Pad shorter sequences to uniform length
//! 5. Create input/target pairs (shift by 1 for autoregressive training)
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use lora_gemma2::tokenizer::GemmaTokenizer;
//! use lora_gemma2::batcher::SFTBatcher;
//! use lora_gemma2::dataset::ChatItem;
//! use burn::data::dataloader::batcher::Batcher;
//!
//! let tokenizer = Arc::new(GemmaTokenizer::from_pretrained("google/gemma-2-2b")?);
//! let batcher = SFTBatcher::new(tokenizer, 2048);
//!
//! let items = vec![dataset.get(0).unwrap()];
//! let batch = batcher.batch(items, &device);
//! // batch.tokens_inputs: [batch, seq_len-1] input tokens
//! // batch.targets:       [batch, seq_len-1] target tokens (shifted by 1)
//! // batch.mask_pad:      [batch, seq_len-1] padding mask (true = ignore)
//! ```

use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::nn::attention::generate_padding_mask;
use burn::prelude::*;

use crate::dataset::ChatItem;
use crate::tokenizer::GemmaTokenizer;

// ---------------------------------------------------------------------------
// Batch Types
// ---------------------------------------------------------------------------

/// Inference batch: padded tokens + padding mask.
///
/// Used for forward passes without training (e.g., evaluation, generation).
#[derive(Debug, Clone)]
pub struct SFTBatch<B: Backend> {
    /// Token IDs: `[batch_size, seq_len]`.
    pub tokens: Tensor<B, 2, Int>,
    /// Padding mask: `[batch_size, seq_len]`.
    /// `true` where tokens are padding (should be ignored).
    pub mask_pad: Tensor<B, 2, Bool>,
}

/// Training batch: input/target pairs for next-token prediction.
///
/// Input tokens are `tokens[:, :-1]` and targets are `tokens[:, 1:]`,
/// implementing the standard autoregressive language modeling objective.
#[derive(Debug, Clone)]
pub struct SFTTrainingBatch<B: Backend> {
    /// Input token IDs: `[batch_size, seq_len - 1]`.
    pub tokens_inputs: Tensor<B, 2, Int>,
    /// Target token IDs: `[batch_size, seq_len - 1]`.
    pub targets: Tensor<B, 2, Int>,
    /// Padding mask for inputs: `[batch_size, seq_len - 1]`.
    /// `true` where tokens are padding (should be ignored in loss).
    pub mask_pad: Tensor<B, 2, Bool>,
}

// ---------------------------------------------------------------------------
// Batcher
// ---------------------------------------------------------------------------

/// Supervised Fine-Tuning batcher.
///
/// Converts [`ChatItem`] conversations into padded tensor batches for
/// Gemma 2 training. Uses the Gemma chat template for formatting.
pub struct SFTBatcher {
    tokenizer: Arc<GemmaTokenizer>,
    max_seq_length: usize,
}

impl SFTBatcher {
    /// Create a new SFT batcher.
    ///
    /// # Arguments
    /// * `tokenizer` — Shared tokenizer instance (wrapped in `Arc` for thread safety)
    /// * `max_seq_length` — Maximum sequence length. Sequences are truncated to this length.
    pub fn new(tokenizer: Arc<GemmaTokenizer>, max_seq_length: usize) -> Self {
        Self {
            tokenizer,
            max_seq_length,
        }
    }

    /// Reference to the underlying tokenizer.
    pub fn tokenizer(&self) -> &GemmaTokenizer {
        &self.tokenizer
    }

    /// Maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Tokenize a single [`ChatItem`] into token IDs using the Gemma chat template.
    ///
    /// Converts the item's messages to [`ChatMessage`]s, formats with the
    /// chat template, and encodes to token IDs. Truncates to `max_seq_length`.
    fn tokenize_item(&self, item: &ChatItem) -> Vec<usize> {
        let messages: Vec<_> = item
            .messages
            .iter()
            .filter_map(|m| m.to_chat_message())
            .collect();

        if messages.is_empty() {
            return vec![self.tokenizer.bos_token_id()];
        }

        let tokens = self.tokenizer.encode_chat(&messages, false);

        // Truncate to max_seq_length
        if tokens.len() > self.max_seq_length {
            tokens[..self.max_seq_length].to_vec()
        } else {
            tokens
        }
    }
}

// ---------------------------------------------------------------------------
// Batcher Trait Implementations
// ---------------------------------------------------------------------------

impl<B: Backend> Batcher<B, ChatItem, SFTBatch<B>> for SFTBatcher {
    fn batch(&self, items: Vec<ChatItem>, device: &B::Device) -> SFTBatch<B> {
        let tokens_list: Vec<Vec<usize>> =
            items.iter().map(|item| self.tokenize_item(item)).collect();

        let mask = generate_padding_mask(
            self.tokenizer.pad_token_id(),
            tokens_list,
            Some(self.max_seq_length),
            device,
        );

        SFTBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<B, ChatItem, SFTTrainingBatch<B>> for SFTBatcher {
    fn batch(&self, items: Vec<ChatItem>, device: &B::Device) -> SFTTrainingBatch<B> {
        // First create an inference batch (padded tokens + mask)
        let inference_batch: SFTBatch<B> =
            Batcher::<B, ChatItem, SFTBatch<B>>::batch(self, items, device);

        let [batch_size, seq_length] = inference_batch.tokens.dims();

        // Ensure seq_length > 1 for the shift operation
        if seq_length <= 1 {
            panic!(
                "Sequence length must be > 1 for next-token prediction, got {seq_length}. \
                 Consider increasing max_seq_length or using longer training samples."
            );
        }

        // Next-token prediction: input = tokens[:-1], target = tokens[1:]
        let tokens_inputs = inference_batch
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = inference_batch.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = inference_batch
            .mask_pad
            .slice([0..batch_size, 0..seq_length - 1]);

        SFTTrainingBatch {
            tokens_inputs,
            targets,
            mask_pad,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ChatMessageSerde;

    /// Helper: create a ChatItem with user/assistant messages.
    fn make_item(user: &str, assistant: &str) -> ChatItem {
        ChatItem {
            messages: vec![
                ChatMessageSerde {
                    role: "user".into(),
                    content: user.into(),
                },
                ChatMessageSerde {
                    role: "assistant".into(),
                    content: assistant.into(),
                },
            ],
        }
    }

    // --- Tokenize tests (no backend needed, string-level) ---

    #[test]
    fn test_batcher_new() {
        // We can't easily test with a real tokenizer without network access,
        // but we can test the struct construction.
        // Real tests use the ndarray backend below.
    }

    // --- Batch dimension tests using ndarray backend ---

    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn test_tokenize_item_empty_messages() {
        // An item with empty messages should produce just BOS
        let item = ChatItem { messages: vec![] };

        // We can't call tokenize_item directly without a real tokenizer,
        // so this test verifies the struct creation path.
        // Full integration test requires a downloaded tokenizer.
        assert!(item.messages.is_empty());
    }

    #[test]
    fn test_training_batch_structure() {
        // Verify the shift operation logic
        // If tokens = [[1, 2, 3, 4]], then:
        //   inputs  = [[1, 2, 3]]
        //   targets = [[2, 3, 4]]
        // This is the standard next-token prediction pattern.
        let device = device();
        let tokens =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::from([[1i64, 2, 3, 4]]), &device);

        let [batch_size, seq_length] = tokens.dims();
        assert_eq!(seq_length, 4);

        let inputs = tokens.clone().slice([0..batch_size, 0..seq_length - 1]);
        let targets = tokens.slice([0..batch_size, 1..seq_length]);

        let input_data: Vec<i64> = inputs.into_data().to_vec().unwrap();
        let target_data: Vec<i64> = targets.into_data().to_vec().unwrap();

        assert_eq!(input_data, vec![1, 2, 3]);
        assert_eq!(target_data, vec![2, 3, 4]);
    }

    #[test]
    fn test_training_batch_multi_sample() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[10i64, 20, 30], [40, 50, 60]]),
            &device,
        );

        let [batch_size, seq_length] = tokens.dims();
        assert_eq!(batch_size, 2);
        assert_eq!(seq_length, 3);

        let inputs = tokens.clone().slice([0..batch_size, 0..seq_length - 1]);
        let targets = tokens.slice([0..batch_size, 1..seq_length]);

        let input_data: Vec<i64> = inputs.into_data().to_vec().unwrap();
        let target_data: Vec<i64> = targets.into_data().to_vec().unwrap();

        assert_eq!(input_data, vec![10, 20, 40, 50]);
        assert_eq!(target_data, vec![20, 30, 50, 60]);
    }

    #[test]
    fn test_mask_shift() {
        // Padding mask should also be shifted consistently
        let device = device();
        let mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[false, false, true, true]]),
            &device,
        );

        let [batch_size, seq_length] = mask.dims();
        let shifted_mask = mask.slice([0..batch_size, 0..seq_length - 1]);

        let mask_data: Vec<bool> = shifted_mask.into_data().to_vec().unwrap();
        // [false, false, true] — first 3 elements
        assert_eq!(mask_data, vec![false, false, true]);
    }

    #[test]
    fn test_chat_item_creation() {
        let item = make_item("Hello", "Hi there!");
        assert_eq!(item.messages.len(), 2);
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[1].role, "assistant");
    }

    #[test]
    fn test_sft_batch_clone() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::ones([2, 4], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::zeros([2, 4], &device);

        let batch = SFTBatch {
            tokens: tokens.clone(),
            mask_pad: mask.clone(),
        };

        let cloned = batch.clone();
        assert_eq!(cloned.tokens.dims(), [2, 4]);
        assert_eq!(cloned.mask_pad.dims(), [2, 4]);
    }

    #[test]
    fn test_training_batch_clone() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::ones([2, 3], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::zeros([2, 3], &device);

        let batch = SFTTrainingBatch {
            tokens_inputs: tokens.clone(),
            targets: tokens.clone(),
            mask_pad: mask,
        };

        let cloned = batch.clone();
        assert_eq!(cloned.tokens_inputs.dims(), [2, 3]);
        assert_eq!(cloned.targets.dims(), [2, 3]);
        assert_eq!(cloned.mask_pad.dims(), [2, 3]);
    }
}
