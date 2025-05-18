//! whisper-stream-rs
//!
//! A library for performing real-time transcription using Whisper ASR models.
//! It handles audio capture, processing, and streaming results.

pub mod audio;
pub mod model;
pub mod stream;
pub mod error;
pub mod audio_utils;
mod score;

pub use stream::{start_transcription_stream, TranscriptionStreamParams, TranscriptionStreamEvent};
// Re-exporting AudioInput for `AudioInput::available_input_devices()` and direct audio capture if needed.
pub use audio::AudioInput;
pub use error::WhisperStreamError;
