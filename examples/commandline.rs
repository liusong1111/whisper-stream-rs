use whisper_stream_rs::stream::{TranscriptionStreamParams, TranscriptionStreamEvent};
use whisper_stream_rs::audio::AudioInput; // For AudioInput::available_input_devices
use whisper_stream_rs::start_transcription_stream;
// use anyhow::Ok; // anyhow::Result implicitly brings Ok into scope for return values
use std::io::{stdout, Write}; // Added for stdout().flush()

fn main() -> anyhow::Result<()> {
    whisper_rs::install_logging_hooks();

    println!("Available audio input devices:");
    let available_devices_result = AudioInput::available_input_devices(); // Call function before match
    match available_devices_result { // Match on the result
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
        // To select a specific device by name (case-insensitive):
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
                    // Ensure a newline before printing final transcript, if a partial one was just printed without one.
                    println!("\r[FINAL] {}\x1b[K", text);
                } else {
                    print!("\r{}\x1b[K", text);
                    if let Err(e) = stdout().flush() {
                        eprintln!("[Error] Failed to flush stdout: {}", e);
                    }
                }
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                println!("\r\x1b[K[SYSTEM] {}", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                eprintln!("\r\x1b[K[ERROR] {}", err);
            }
        }
    }
    Ok(())
}