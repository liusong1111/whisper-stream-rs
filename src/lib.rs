//! whisper-stream-rs
//!
//! A library for performing real-time transcription using Whisper ASR models.
//! It handles audio capture, processing, and streaming results.

mod audio;
mod audio_utils;
mod error;
mod model;
mod score;
mod whisper_stream;
// New public API
pub use error::WhisperStreamError;
pub use model::{DEFAULT_MODEL, ensure_model};
pub use whisper_stream::{Event, WhisperStream};
