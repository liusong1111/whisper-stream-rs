use whisper_stream_rs::{
    start_transcription_stream, AudioInput, TranscriptionStreamEvent, TranscriptionStreamParams,
    WhisperStreamError, calculate_score, install_logging_hooks,
};
use std::{
    io::{stdout, Write},
    path::PathBuf,
    fs,
    time::Duration,
    thread,
};

// Helper function to create a directory if it doesn't exist
fn ensure_dir_exists(path_str: &str) -> anyhow::Result<()> {
    let path = PathBuf::from(path_str);
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
            println!("[Setup] Created directory: {}", parent.display());
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // --- Initialization ---
    // Optional: Initialize a logger like env_logger to see internal logs.
    // For detailed logs, you might set RUST_LOG=whisper_stream_rs=debug,whisper_rs=info
    // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,whisper_stream_rs=debug,whisper_rs=debug")).init();
    // If you don't initialize a logger, whisper.cpp logs will go to stdout/stderr by default
    // after calling install_logging_hooks().
    install_logging_hooks();
    println!("[Maximum Example] whisper-stream-rs comprehensive features demo.");
    println!("[Maximum Example] Press Ctrl+C to stop sections that involve active listening.\n");

    // --- 1. Audio Device Management ---
    println!("--- Section 1: Audio Device Listing ---");
    match AudioInput::available_input_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("[Audio Devices] No input devices found.");
            } else {
                println!("[Audio Devices] Available input devices:");
                for (i, name) in devices.iter().enumerate() {
                    println!("  {}: {}", i, name);
                }
                // For the example, we'll try to use the first available device, or default if none.
                // In a real app, you might let the user choose.
            }
        }
        Err(e) => {
            eprintln!("[Audio Devices] Error listing input devices: {}", e);
        }
    }
    // Let's pick the first device if available, otherwise use None for default.
    // This is just for demonstration; robust applications would handle this choice more gracefully.
    let preferred_device_name = AudioInput::available_input_devices()
        .ok()
        .and_then(|devices| devices.into_iter().next());

    if let Some(ref name) = preferred_device_name {
        println!("[Audio Devices] Will attempt to use device: '{}' for some examples.\n", name);
    } else {
        println!("[Audio Devices] No specific device found/selected, will use system default.\n");
    }


    // --- 2. Demonstrate `calculate_score` ---
    println!("--- Section 2: `calculate_score` Demonstration ---");
    let test_texts = [
        "Hello world, this is a normal sentence.",
        "[silence]",
        "[ Speaking ]", // Note: space sensitive for `starts_with` after trim
        "(various noises)",
        "Just a regular text.",
        "[BLANK_AUDIO] this should be high score",
        "Another one (with parentheses)",
    ];
    for &text in test_texts.iter() {
        let score = calculate_score(text);
        println!("[Score Demo] Text: '{}' -> Score: {}", text, score);
    }
    println!("\n");


    // --- 3. Transcription with Custom Parameters (including WAV recording) ---
    println!("--- Section 3: Transcription with Custom Params & WAV Recording ---");
    println!("[Info] This section will record audio to 'recordings/max_demo_custom.wav'");
    println!("[Info] Speak for a few seconds (e.g., 10-15s). Press Ctrl+C to stop early.\n");

    let wav_output_path_custom = "recordings/max_demo_custom.wav";
    ensure_dir_exists(wav_output_path_custom)?;

    let custom_params = TranscriptionStreamParams {
        record_to_wav: Some(wav_output_path_custom.to_string()),
        language: Some("en".to_string()), // Specify English
        step_ms: 750,                     // Process audio in 750ms chunks
        length_ms: 6000,                  // 6-second transcription window
        keep_ms: 300,                     // Keep 300ms for context
        max_tokens: 48,                   // Allow slightly more tokens
        n_threads: 2, // Use 2 threads (adjust based on your CPU, or use default)
        audio_device_name: preferred_device_name.clone(), // Use the selected device or default
        compute_partials: true,           // Get live updates
    };

    println!("[Custom Params] Using parameters: {:#?}", custom_params);
    let rx_custom = start_transcription_stream(custom_params);
    let stream_duration = Duration::from_secs(15); // Let it run for 15 seconds
    let start_time = std::time::Instant::now();

    for event in rx_custom {
        if start_time.elapsed() > stream_duration {
            println!("\n[Custom Params] Desired duration reached. Stopping this section.");
            // To properly stop the stream and finalize WAV, the thread inside start_transcription_stream
            // needs to exit. Dropping the receiver `rx_custom` signals this.
            // However, the internal thread might take a moment to shut down.
            // For this example, we break the loop. In a real app, you'd manage stream lifetime.
            break;
        }
        match event {
            TranscriptionStreamEvent::ProvisionalLiveUpdate { text, score } => {
                print!("\r[Custom Provisional] (Score: {}) {}[K", score, text);
                let _ = stdout().flush();
            }
            TranscriptionStreamEvent::SegmentTranscript { text, score } => {
                println!("\r[Custom Segment] (Score: {}) {}[K", score, text);
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                println!("\r[Custom System] {}[K", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                eprintln!("\r[Custom Error] {}[K", err);
                match err {
                    WhisperStreamError::ModelLoad(_) => {
                        eprintln!("Hint: Ensure the Whisper model (e.g., ggml-base.en.bin) is downloaded.");
                        eprintln!("It should typically be in ~/.local/share/whisper-stream-rs/");
                        return Err(anyhow::Error::new(err)); // Critical error
                    }
                    WhisperStreamError::AudioDevice(msg) => {
                         eprintln!("Audio device error: {}. Check microphone permissions and device selection.", msg);
                         // Depending on severity, might not be fatal for the whole example run.
                    }
                    // Handle other specific errors as needed
                    _ => {}
                }
            }
        }
    }
    // Give a moment for the recording to finalize if it was abruptly stopped.
    // Proper shutdown would involve signalling the stream thread.
    thread::sleep(Duration::from_millis(500));
    println!("[Custom Params] Finished section. Check {} if recording was active.\n", wav_output_path_custom);


    // --- 4. Transcription with Auto Language Detection & No Partials ---
    println!("--- Section 4: Auto Language Detection & No Partials ---");
    println!("[Info] Speak in any supported language for a few seconds (e.g., 10-15s). No live updates.");
    println!("[Info] Audio will be recorded to 'recordings/max_demo_auto_lang.wav'\n");

    let wav_output_path_auto = "recordings/max_demo_auto_lang.wav";
    ensure_dir_exists(wav_output_path_auto)?;

    let auto_lang_params = TranscriptionStreamParams {
        record_to_wav: Some(wav_output_path_auto.to_string()),
        language: None, // Auto-detect language
        step_ms: 1000,
        length_ms: 7000,
        keep_ms: 500,
        max_tokens: 32, // Default
        n_threads: std::thread::available_parallelism().map_or(2, |n| n.get() as i32), // Use available parallelism
        audio_device_name: None, // Use default device
        compute_partials: false, // No provisional updates
    };

    println!("[Auto Lang Params] Using parameters: {:#?}", auto_lang_params);
    let rx_auto_lang = start_transcription_stream(auto_lang_params);
    let stream_duration_auto = Duration::from_secs(15);
    let start_time_auto = std::time::Instant::now();

    for event in rx_auto_lang {
         if start_time_auto.elapsed() > stream_duration_auto {
            println!("\n[Auto Lang Params] Desired duration reached. Stopping this section.");
            break;
        }
        match event {
            TranscriptionStreamEvent::ProvisionalLiveUpdate { .. } => {
                // This should not be received if compute_partials is false
                println!("\r[Auto Lang] Unexpected Provisional Update![K");
            }
            TranscriptionStreamEvent::SegmentTranscript { text, score } => {
                println!("\r[Auto Lang Segment] (Score: {}) {}[K", score, text);
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                println!("\r[Auto Lang System] {}[K", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                eprintln!("\r[Auto Lang Error] {}[K", err);
            }
        }
    }
    thread::sleep(Duration::from_millis(500)); // Allow for WAV finalization
    println!("[Auto Lang Params] Finished section. Check {} if recording was active.\n", wav_output_path_auto);


    // --- 5. Brief Transcription with different timing ---
    println!("--- Section 5: Short Transcription with Fast Updates (minimal latency focus) ---");
    println!("[Info] Speak briefly. This aims for quicker, possibly less accurate, partials.");
    println!("[Info] No WAV recording for this one.\n");

    let fast_params = TranscriptionStreamParams {
        record_to_wav: None, // No recording
        language: Some("en".to_string()),
        step_ms: 250,                     // Very short step
        length_ms: 2000,                  // Short window
        keep_ms: 100,                     // Minimal context
        max_tokens: 16,
        n_threads: 2,
        audio_device_name: None,
        compute_partials: true,
    };

    println!("[Fast Params] Using parameters: {:#?}", fast_params);
    let rx_fast = start_transcription_stream(fast_params);
    let stream_duration_fast = Duration::from_secs(8);
    let start_time_fast = std::time::Instant::now();

    for event in rx_fast {
        if start_time_fast.elapsed() > stream_duration_fast {
            println!("\n[Fast Params] Desired duration reached. Stopping this section.");
            break;
        }
        match event {
            TranscriptionStreamEvent::ProvisionalLiveUpdate { text, score } => {
                print!("\r[Fast Provisional] (Score: {}) {}[K", score, text);
                let _ = stdout().flush();
            }
            TranscriptionStreamEvent::SegmentTranscript { text, score } => {
                println!("\r[Fast Segment] (Score: {}) {}[K", score, text);
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                println!("\r[Fast System] {}[K", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                eprintln!("\r[Fast Error] {}[K", err);
            }
        }
    }
    println!("[Fast Params] Finished section.\n");


    println!("[Maximum Example] All sections completed. Exiting.");
    Ok(())
}