use whisper_stream_rs::{
    start_transcription_stream, TranscriptionStreamEvent, TranscriptionStreamParams,
    install_logging_hooks,
};
use std::io::{stdout, Write}; // For flushing stdout

fn main() -> anyhow::Result<()> {
    // Optional: Initialize a logger like env_logger to see internal logs from whisper-stream-rs and whisper_rs.
    // Example: env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // This is important to redirect whisper.cpp logs (used by whisper-rs)
    // to the `log` facade or to stdout/stderr if no logger is explicitly configured.
    install_logging_hooks();

    println!("[Minimal Example] Starting transcription with default parameters...");
    println!("[Minimal Example] Speak into your microphone. Press Ctrl+C to stop.\n");

    let params = TranscriptionStreamParams::default();
    let rx = start_transcription_stream(params);

    for event in rx {
        match event {
            TranscriptionStreamEvent::ProvisionalLiveUpdate { text, score } => {
                // Overwrite the current line for live updates, clear rest of line with \x1b[K
                print!("\r[Provisional] (Score: {}) {}\x1b[K", score, text);
                let _ = stdout().flush(); // Ensure the update is displayed immediately
            }
            TranscriptionStreamEvent::SegmentTranscript { text, score } => {
                // Clear the current line (which might have a provisional transcript)
                // and print the final segment, followed by a newline.
                println!("\r[Segment] (Score: {}) {}\x1b[K", score, text);
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                // Ensure system messages clear any provisional text and start on a new line if needed.
                println!("\r[System] {}\x1b[K", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                // Ensure errors clear any provisional text and start on a new line.
                eprintln!("\r[Error] An error occurred: {}\x1b[K", err);
                // Consider breaking the loop or handling the error more robustly depending on needs.
            }
        }
    }

    Ok(())
}