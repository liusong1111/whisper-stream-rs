use std::path::{Path, PathBuf};
use std::fs;
use std::io::{self, Write};

const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
const MODEL_FILENAME: &str = "ggml-base.en.bin";

/// Ensures the Whisper model is present, downloading if necessary.
pub fn ensure_model() -> io::Result<PathBuf> {
    let cache_dir = dirs::data_local_dir()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Could not find local data dir"))?
        .join("whisper-stream-rs");
    fs::create_dir_all(&cache_dir)?;
    let model_path = cache_dir.join(MODEL_FILENAME);
    if !model_path.exists() {
        println!("Downloading Whisper model to {}...", model_path.display());
        let mut resp = reqwest::blocking::get(MODEL_URL)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Download failed: {e}")))?;
        let mut out = fs::File::create(&model_path)?;
        io::copy(&mut resp, &mut out)?;
        out.flush()?;
        println!("Model downloaded.");
    }
    Ok(model_path)
}