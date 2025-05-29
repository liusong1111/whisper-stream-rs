use whisper_stream_rs::{WhisperStream, Event};
use std::io::{stdout, Write}; // For flushing stdout

fn main() -> anyhow::Result<()> {
    // Optional: Initialize a logger like env_logger to see internal logs from whisper-stream-rs and whisper_rs.
    // Example: env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("[Minimal Example] Starting transcription with default parameters...");
    println!("[Minimal Example] Speak into your microphone. Press Ctrl+C to stop.\n");

    let (_stream, rx) = WhisperStream::builder().build()?;

    for event in rx {
        match event {
            Event::ProvisionalLiveUpdate { text, is_low_quality } => {
                // Overwrite the current line for live updates, clear rest of line with \x1b[K
                print!("\r[Provisional] (Low Quality: {}) {}\x1b[K", is_low_quality, text);
                let _ = stdout().flush(); // Ensure the update is displayed immediately
            }
            Event::SegmentTranscript { text, is_low_quality } => {
                // Clear the current line (which might have a provisional transcript)
                // and print the final segment, followed by a newline.
                println!("\r[Segment] (Low Quality: {}) {}\x1b[K", is_low_quality, text);
            }
            Event::SystemMessage(msg) => {
                // Ensure system messages clear any provisional text and start on a new line if needed.
                println!("\r[System] {}\x1b[K", msg);
            }
            Event::Error(err) => {
                // Ensure errors clear any provisional text and start on a new line.
                eprintln!("\r[Error] An error occurred: {}\x1b[K", err);
                // Consider breaking the loop or handling the error more robustly depending on needs.
            }
        }
    }

    Ok(())
}