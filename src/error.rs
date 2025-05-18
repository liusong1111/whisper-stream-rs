use thiserror::Error;
use cpal::{BuildStreamError, DefaultStreamConfigError, DevicesError, StreamError as CpalStreamError};

/// Represents all possible errors that can occur within the `whisper-stream-rs` library.
#[derive(Error, Debug)]
pub enum WhisperStreamError {
    #[error("Failed to load or initialize the Whisper model: {0}")]
    ModelLoad(String),

    #[error("General audio input/initialization error: {0}")]
    AudioInit(String),

    #[error("Audio device error: {0}")]
    AudioDevice(String),

    #[error("Audio stream configuration error: {0}")]
    AudioStreamConfig(String),

    #[error("Audio stream creation error: {0}")]
    AudioStreamCreation(String),

    #[error("Audio stream runtime error: {0}")]
    AudioStreamRuntime(String),

    #[error("Audio resampling error: {0}")]
    AudioResampling(String),

    #[error("Failed to create or manage Whisper context/state: {0}")]
    Context(String),

    #[error("Error during audio transcription: {0}")]
    Transcription(String),

    #[error("Failed to write audio to WAV file: {0}")]
    WavWrite(String),

    #[error("An internal library error occurred: {0}")]
    Internal(String),

    #[error("Failed to retrieve or download model: {0}")]
    ModelFetch(String),

    #[error("I/O error: {source}")]
    Io { #[from] source: std::io::Error },

    #[error("Hound (WAV processing) error: {source}")]
    Hound { #[from] source: hound::Error },

    #[error("Whisper context error: {source}")]
    WhisperContext { #[from] source: whisper_rs::WhisperError },

    #[error("CPAL device enumeration error: {source}")]
    CpalDevicesError { #[from] source: DevicesError },

    #[error("CPAL default stream config error: {source}")]
    CpalDefaultConfigError { #[from] source: DefaultStreamConfigError },

    #[error("CPAL build stream error: {source}")]
    CpalBuildStreamError { #[from] source: BuildStreamError },

    #[error("CPAL runtime stream error: {0}")]
    CpalRuntimeStreamError(String),

    #[error("Reqwest HTTP client error: {source}")]
    ReqwestError{ #[from] source: reqwest::Error },
}

// Manual conversion for CpalStreamError as it's an enum and needs specific handling
impl From<CpalStreamError> for WhisperStreamError {
    fn from(err: CpalStreamError) -> Self {
        WhisperStreamError::CpalRuntimeStreamError(err.to_string())
    }
}