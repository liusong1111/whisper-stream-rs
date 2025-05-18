use std::path::{PathBuf};
use std::fs;
use std::io::{self, Write};
use crate::error::WhisperStreamError;

const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
const MODEL_FILENAME: &str = "ggml-base.en.bin";

/// Ensures the Whisper model is present, downloading if necessary.
pub fn ensure_model() -> Result<PathBuf, WhisperStreamError> {
    let cache_dir = dirs::data_local_dir()
        .ok_or_else(|| WhisperStreamError::Io {
            source: io::Error::new(io::ErrorKind::NotFound, "Could not find local data dir")
        })?
        .join("whisper-stream-rs");

    fs::create_dir_all(&cache_dir).map_err(WhisperStreamError::from)?;

    let model_path = cache_dir.join(MODEL_FILENAME);

    if !model_path.exists() {
        println!("Downloading Whisper model to {}...", model_path.display());

        let mut resp = reqwest::blocking::get(MODEL_URL)
            .map_err(WhisperStreamError::from)?;

        let mut out = fs::File::create(&model_path).map_err(WhisperStreamError::from)?;

        io::copy(&mut resp, &mut out).map_err(WhisperStreamError::from)?;

        out.flush().map_err(WhisperStreamError::from)?;

        println!("Model downloaded.");
    }
    Ok(model_path)
}