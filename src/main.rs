use whisper_stream_rs::stream::{TranscriptionStreamParams, TranscriptionStreamEvent};
use whisper_stream_rs::audio::AudioInput;
use whisper_stream_rs::start_transcription_stream;

fn main() -> anyhow::Result<()> {
    whisper_rs::install_logging_hooks();

    // Optional: List available audio devices
    println!("Available audio input devices:");
    match AudioInput::available_input_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("  No input devices found.");
            } else {
                for (i, name) in devices.iter().enumerate() {
                    println!("  {}: {}", i, name);
                }
            }
        }
        Err(e) => {
            eprintln!("[Warning] Could not list audio devices: {}", e);
        }
    }
    println!("---");

    let params = TranscriptionStreamParams {
        record_to_wav: Some("test_recording.wav".to_string()),
        // To select a specific device by name:
        // audio_device_name: Some("Your Device Name Here".to_string()),
        ..Default::default()
    };

    println!(
        "[Config] Using audio device: {}",
        params.audio_device_name.as_deref().unwrap_or("Default System Device")
    );
    println!("[Config] Step (chunk) duration: {}ms", params.step_ms);
    println!("[Config] Language: {}", params.language.as_deref().unwrap_or("auto"));
    println!("[Config] Recording to WAV: {}", params.record_to_wav.as_deref().unwrap_or("No"));
    println!("---");


    let rx = start_transcription_stream(params);

    println!("[System] Start speaking...");

    for event in rx {
        match event {
            TranscriptionStreamEvent::Transcript { text, is_final } => {
                if is_final {
                    println!("[FINAL] {}", text);
                } else {
                    // For non-final, clear the line and reprint, or just print continuously
                    // depending on preference for live updates.
                    // Simple continuous print for now:
                    print!("\r{}\x1b[K", text); // Clear line, print text.
                    // For more robust TUI, consider libraries like crossterm.
                    use std::io::{stdout, Write};
                    stdout().flush().unwrap(); // Important to make print! effective with
                }
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                // Ensure newline after potentially incomplete live transcript
                println!("\r\x1b[K[SYSTEM] {}", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                // Ensure newline after potentially incomplete live transcript
                eprintln!("\r\x1b[K[ERROR] {}", err);
            }
        }
    }
    Ok(())
}