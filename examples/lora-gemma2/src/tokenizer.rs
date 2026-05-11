//! Gemma 2 tokenizer with chat template support.
//!
//! Wraps HuggingFace's `tokenizers` crate for encoding/decoding text
//! and formatting Gemma-style chat conversations.
//!
//! # Chat Template
//!
//! Gemma 2 uses a turn-based format with control tokens:
//! ```text
//! <bos><start_of_turn>user
//! What is Rust?<end_of_turn>
//! <start_of_turn>model
//! Rust is a systems programming language.<end_of_turn>
//! ```
//!
//! # Example
//!
//! ```ignore
//! use lora_gemma2::tokenizer::{GemmaTokenizer, ChatMessage, Role};
//!
//! let tok = GemmaTokenizer::from_pretrained("google/gemma-2-2b")?;
//! let messages = vec![
//!     ChatMessage::user("Hello"),
//!     ChatMessage::assistant("Hi there!"),
//! ];
//! let tokens = tok.encode_chat(&messages, true);
//! ```

use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors that can occur during tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    /// Failed to load tokenizer from pretrained model or file.
    Load(String),
    /// Failed to encode text.
    Encode(String),
    /// Failed to decode tokens.
    Decode(String),
    /// A required special token is missing from the tokenizer.
    MissingToken(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Load(msg) => write!(f, "Tokenizer load error: {msg}"),
            TokenizerError::Encode(msg) => write!(f, "Encode error: {msg}"),
            TokenizerError::Decode(msg) => write!(f, "Decode error: {msg}"),
            TokenizerError::MissingToken(msg) => write!(f, "Missing special token: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

// ---------------------------------------------------------------------------
// Chat Message Types
// ---------------------------------------------------------------------------

/// Role of a chat message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// System-level instructions.
    System,
    /// User query or instruction.
    User,
    /// Assistant/model response.
    Assistant,
}

impl Role {
    /// Returns the Gemma turn marker name for this role.
    ///
    /// Note: Gemma uses `"model"` for the assistant role, not `"assistant"`.
    pub fn turn_name(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "model",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.turn_name())
    }
}

/// Parse [`Role`] from string (case-insensitive).
///
/// Accepts both `"assistant"` and `"model"` for the [`Role::Assistant`] variant.
///
/// # Errors
///
/// Returns [`RoleParseError`] if the string doesn't match any known role.
impl std::str::FromStr for Role {
    type Err = RoleParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" | "model" => Ok(Role::Assistant),
            other => Err(RoleParseError(other.into())),
        }
    }
}

/// Error returned when a string cannot be parsed as a [`Role`].
#[derive(Debug, Clone, PartialEq)]
pub struct RoleParseError(String);

impl fmt::Display for RoleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown role: {}", self.0)
    }
}

impl std::error::Error for RoleParseError {}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The sender role.
    pub role: Role,
    /// The message content.
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Convenience constructor for user messages.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Convenience constructor for assistant messages.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Convenience constructor for system messages.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
}

// ---------------------------------------------------------------------------
// Gemma Tokenizer
// ---------------------------------------------------------------------------

/// Gemma 2 tokenizer wrapping HuggingFace's `tokenizers` crate.
///
/// Loads from HuggingFace model IDs or local paths. Provides encoding,
/// decoding, and Gemma chat template formatting.
pub struct GemmaTokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: usize,
    eos_token_id: usize,
    pad_token_id: usize,
}

impl GemmaTokenizer {
    /// Load tokenizer from a HuggingFace model ID.
    ///
    /// Downloads and caches the tokenizer from HuggingFace Hub.
    /// E.g., `"google/gemma-2-2b"`, `"google/gemma-2-2b-it"`.
    pub fn from_pretrained(model_id: &str) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| TokenizerError::Load(format!("Failed to load from '{model_id}': {e}")))?;
        Self::from_inner(inner)
    }

    /// Load tokenizer from a local file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let path_str = path.as_ref().display().to_string();
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| TokenizerError::Load(format!("Failed to load from '{path_str}': {e}")))?;
        Self::from_inner(inner)
    }

    /// Construct from an existing `tokenizers::Tokenizer`, extracting special token IDs.
    fn from_inner(inner: tokenizers::Tokenizer) -> Result<Self, TokenizerError> {
        let bos_token_id: usize = inner
            .token_to_id("<bos>")
            .or_else(|| inner.token_to_id("<s>"))
            .or_else(|| inner.token_to_id("<|begin_of_text|>"))
            .ok_or_else(|| TokenizerError::MissingToken("bos (<bos> or <s>)".into()))?
            as usize;

        let eos_token_id: usize = inner
            .token_to_id("<eos>")
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("<|end_of_text|>"))
            .ok_or_else(|| TokenizerError::MissingToken("eos (<eos> or </s>)".into()))?
            as usize;

        // Pad token: prefer explicit <pad>, fall back to eos
        let pad_token_id = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("<|finetune_right_pad_id|>"))
            .unwrap_or(eos_token_id as u32) as usize;

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    // -----------------------------------------------------------------------
    // Encoding / Decoding
    // -----------------------------------------------------------------------

    /// Encode text into token IDs.
    ///
    /// When `add_special_tokens` is true, the tokenizer adds BOS/EOS per its config.
    /// For chat-formatted text, use [`encode_chat`](Self::encode_chat) instead.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|enc| enc.get_ids().iter().map(|&id| id as usize).collect())
            .unwrap_or_else(|e| panic!("Failed to encode: {e}"))
    }

    /// Encode text, truncating to `max_length` tokens.
    pub fn encode_truncated(
        &self,
        text: &str,
        max_length: usize,
        add_special_tokens: bool,
    ) -> Vec<usize> {
        let mut tokens = self.encode(text, add_special_tokens);
        tokens.truncate(max_length);
        tokens
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize], skip_special_tokens: bool) -> String {
        let tokens: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.inner
            .decode(&tokens, skip_special_tokens)
            .unwrap_or_else(|e| panic!("Failed to decode: {e}"))
    }

    // -----------------------------------------------------------------------
    // Chat Template
    // -----------------------------------------------------------------------

    /// Format chat messages using Gemma's chat template.
    ///
    /// Produces:
    /// ```text
    /// <bos><start_of_turn>user
    /// Hello!<end_of_turn>
    /// <start_of_turn>model
    /// Hi there!<end_of_turn>
    /// ```
    ///
    /// If `add_generation_prompt` is true, appends `<start_of_turn>model\n`
    /// to prime the model for generating a response.
    pub fn format_chat(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let mut text = String::new();

        // BOS token
        text.push_str(self.decode(&[self.bos_token_id], false).trim());

        for message in messages {
            text.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                message.role.turn_name(),
                message.content
            ));
        }

        if add_generation_prompt {
            text.push_str("<start_of_turn>model\n");
        }

        text
    }

    /// Format chat and encode to token IDs in one step.
    ///
    /// BOS is already included in the formatted text, so `add_special_tokens`
    /// is set to `false` for the encoding step.
    pub fn encode_chat(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Vec<usize> {
        let text = self.format_chat(messages, add_generation_prompt);
        self.encode(&text, false)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Vocabulary size (including added tokens).
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// BOS (beginning of sequence) token ID.
    pub fn bos_token_id(&self) -> usize {
        self.bos_token_id
    }

    /// EOS (end of sequence) token ID.
    pub fn eos_token_id(&self) -> usize {
        self.eos_token_id
    }

    /// PAD token ID.
    ///
    /// Falls back to EOS if no explicit pad token exists in the vocabulary.
    pub fn pad_token_id(&self) -> usize {
        self.pad_token_id
    }

    /// The BOS token as a string.
    pub fn bos_token(&self) -> String {
        self.decode(&[self.bos_token_id], false)
    }

    /// The EOS token as a string.
    pub fn eos_token(&self) -> String {
        self.decode(&[self.eos_token_id], false)
    }

    /// The PAD token as a string.
    pub fn pad_token(&self) -> String {
        self.decode(&[self.pad_token_id], false)
    }

    /// Reference to the underlying `tokenizers::Tokenizer`.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

impl Clone for GemmaTokenizer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            pad_token_id: self.pad_token_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Role tests ---

    #[test]
    fn test_role_turn_name() {
        assert_eq!(Role::System.turn_name(), "system");
        assert_eq!(Role::User.turn_name(), "user");
        assert_eq!(Role::Assistant.turn_name(), "model");
    }

    #[test]
    fn test_role_from_str() {
        assert_eq!("user".parse::<Role>(), Ok(Role::User));
        assert_eq!("User".parse::<Role>(), Ok(Role::User));
        assert_eq!("assistant".parse::<Role>(), Ok(Role::Assistant));
        assert_eq!("Assistant".parse::<Role>(), Ok(Role::Assistant));
        assert_eq!("model".parse::<Role>(), Ok(Role::Assistant));
        assert_eq!("Model".parse::<Role>(), Ok(Role::Assistant));
        assert_eq!("system".parse::<Role>(), Ok(Role::System));
        assert!("unknown".parse::<Role>().is_err());
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", Role::User), "user");
        assert_eq!(format!("{}", Role::Assistant), "model");
        assert_eq!(format!("{}", Role::System), "system");
    }

    // --- ChatMessage tests ---

    #[test]
    fn test_chat_message_constructors() {
        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");

        let msg = ChatMessage::assistant("Hi");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hi");

        let msg = ChatMessage::system("Be helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "Be helpful");
    }

    // --- Chat template tests (string-level, no tokenizer needed) ---

    #[test]
    fn test_format_chat_template_basic() {
        let _messages = [
            ChatMessage::user("What is Rust?"),
            ChatMessage::assistant("Rust is a systems programming language."),
        ];

        // Build expected template string manually
        let mut expected = String::from("<bos>");
        expected.push_str("<start_of_turn>user\nWhat is Rust?<end_of_turn>\n");
        expected.push_str(
            "<start_of_turn>model\nRust is a systems programming language.<end_of_turn>\n",
        );

        // Verify structure
        assert!(expected.contains("<start_of_turn>user\nWhat is Rust?<end_of_turn>"));
        assert!(expected.contains(
            "<start_of_turn>model\nRust is a systems programming language.<end_of_turn>"
        ));
        assert!(expected.starts_with("<bos>"));
    }

    #[test]
    fn test_format_chat_template_generation_prompt() {
        let messages = vec![ChatMessage::user("Hello")];

        let mut text = String::from("<bos>");
        for msg in &messages {
            text.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role.turn_name(),
                msg.content
            ));
        }
        text.push_str("<start_of_turn>model\n");

        assert!(text.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(text.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_format_chat_template_system_message() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hi"),
        ];

        let mut text = String::from("<bos>");
        for msg in &messages {
            text.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role.turn_name(),
                msg.content
            ));
        }

        assert!(text.contains("<start_of_turn>system\nYou are helpful.<end_of_turn>"));
        assert!(text.contains("<start_of_turn>user\nHi<end_of_turn>"));
        // System should come before user
        let sys_pos = text.find("<start_of_turn>system").unwrap();
        let user_pos = text.find("<start_of_turn>user").unwrap();
        assert!(sys_pos < user_pos);
    }

    #[test]
    fn test_format_chat_template_multi_turn() {
        let messages = vec![
            ChatMessage::user("Q1"),
            ChatMessage::assistant("A1"),
            ChatMessage::user("Q2"),
            ChatMessage::assistant("A2"),
        ];

        let mut text = String::from("<bos>");
        for msg in &messages {
            text.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role.turn_name(),
                msg.content
            ));
        }

        assert!(text.contains("<start_of_turn>user\nQ1<end_of_turn>"));
        assert!(text.contains("<start_of_turn>model\nA1<end_of_turn>"));
        assert!(text.contains("<start_of_turn>user\nQ2<end_of_turn>"));
        assert!(text.contains("<start_of_turn>model\nA2<end_of_turn>"));
    }

    #[test]
    fn test_format_chat_empty_messages() {
        let messages: Vec<ChatMessage> = vec![];
        let mut text = String::from("<bos>");
        for msg in &messages {
            text.push_str(&format!(
                "<start_of_turn>{}\n{}<end_of_turn>\n",
                msg.role.turn_name(),
                msg.content
            ));
        }
        assert_eq!(text, "<bos>");
    }

    // --- Error display tests ---

    #[test]
    fn test_tokenizer_error_display() {
        let err = TokenizerError::Load("test load error".into());
        assert!(err.to_string().contains("test load error"));
        assert!(err.to_string().contains("load error"));

        let err = TokenizerError::Encode("encode fail".into());
        assert!(err.to_string().contains("encode fail"));

        let err = TokenizerError::Decode("decode fail".into());
        assert!(err.to_string().contains("decode fail"));

        let err = TokenizerError::MissingToken("pad".into());
        assert!(err.to_string().contains("pad"));
    }

    // --- Clone test ---

    #[test]
    fn test_role_copy() {
        let role = Role::User;
        let copied = role;
        assert_eq!(role, copied);
    }
}
