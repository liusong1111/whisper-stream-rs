use whisper_stream_rs::{WhisperStream, Event};
use std::io::{stdout, Write};

fn main() -> anyhow::Result<()> {
    // Show available audio devices and models
    println!("Available devices: {:?}", WhisperStream::list_devices()?);
    println!("Available models: {:?}", WhisperStream::list_models());

    // Hardcoded config: record to WAV, set language, adjust params
    let (_stream, rx) = WhisperStream::builder()
        .language("en")
        .record_to_wav("recorded_audio.wav")
        .step_ms(500)
        .length_ms(4000)
        .keep_ms(300)
        .n_threads(8)
        .build()?;

    println!("[Standart Example] Speak into your microphone. Press Ctrl+C to stop.\n");

    let mut prev_provisional_low_quality = false;
    for event in rx {
        match event {
            Event::ProvisionalLiveUpdate { text, is_low_quality } => {
                if is_low_quality && prev_provisional_low_quality {
                    // Skip printing if both current and previous are low quality
                    continue;
                }
                print!("\r[Provisional] (Low Quality: {}) {}\x1b[K", is_low_quality, text);
                let _ = stdout().flush();
                prev_provisional_low_quality = is_low_quality;
            }
            Event::SegmentTranscript { text, is_low_quality } => {
                println!("\r[Segment] (Low Quality: {}) {}\x1b[K", is_low_quality, text);
                prev_provisional_low_quality = false; // Reset on segment
            }
            Event::SystemMessage(msg) => {
                println!("\r[System] {}\x1b[K", msg);
            }
            Event::Error(err) => {
                eprintln!("\r[Error] An error occurred: {}\x1b[K", err);
            }
        }
    }
    Ok(())
}