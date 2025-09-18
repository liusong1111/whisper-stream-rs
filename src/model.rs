use crate::error::WhisperStreamError;
use std::path::PathBuf;
use std::str::FromStr;

pub const DEFAULT_MODEL: &str = "ggml-large-v3-turbo-q5_0.bin";

/// Ensures the Whisper model (and CoreML model if 'coreml' feature is enabled) is present, downloading if necessary.
pub fn ensure_model(model_path: &str) -> Result<PathBuf, WhisperStreamError> {
    let model_path = PathBuf::from_str(model_path).map_err(|err| {
        WhisperStreamError::ModelFetch(format!(
            "can not find model, model_path={}, err={}",
            model_path, err
        ))
    })?;
    Ok(model_path) // Return path to the main .bin model
}
