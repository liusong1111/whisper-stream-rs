use whisper_stream_rs::stream::{TranscriptionStreamParams, TranscriptionStreamEvent};
use whisper_stream_rs::audio::AudioInput; // For AudioInput::available_input_devices
use whisper_stream_rs::start_transcription_stream;
use std::io::{stdout, Write}; // Added for stdout().flush()
use env_logger;

fn main() -> anyhow::Result<()> {
    // Initialize env_logger. Logs can be controlled by RUST_LOG environment variable.
    // e.g., RUST_LOG=whisper_stream_rs=debug,whisper_rs=info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // This is important to redirect whisper.cpp logs to the `log` facade
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
                // err is now WhisperStreamError
                // We can use anyhow to add context for the command-line application
                let app_error = anyhow::Error::new(err).context("An error occurred in the transcription stream");
                eprintln!("\r\x1b[K[ERROR] {:#}", app_error); // Use {:#} for multi-line display with anyhow
            }
        }
    }
    Ok(())
}