use whisper_stream_rs::stream::{TranscriptionStreamParams, TranscriptionStreamEvent};
use whisper_stream_rs::audio::AudioInput; // For AudioInput::available_input_devices
use whisper_stream_rs::{start_transcription_stream, install_logging_hooks};
use std::io::{stdout, Write}; // Added for stdout().flush()
use env_logger;
use clap::Parser;

/// Command-line tool to stream audio from a microphone and transcribe it using whisper-stream-rs.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    /// Target language for transcription (e.g., "en", "es"). If not set, defaults to auto-detection.
    #[clap(short, long)]
    language: Option<String>,

    /// Duration of each audio chunk processed by AudioInput in milliseconds.
    #[clap(long)]
    step_ms: Option<u32>,

    /// Total duration of the audio window considered for a single transcription in milliseconds.
    #[clap(long)]
    length_ms: Option<u32>,

    /// Duration of audio from the previous segment to keep for context in milliseconds.
    #[clap(long)]
    keep_ms: Option<u32>,

    /// Maximum number of tokens to generate per audio segment.
    #[clap(long)]
    max_tokens: Option<i32>,

    /// Number of threads to use for Whisper model computation. Defaults to available parallelism.
    #[clap(long)]
    n_threads: Option<i32>,

    /// Name of the audio input device to use (case-insensitive). Uses system default if not specified.
    #[clap(long)]
    audio_device_name: Option<String>,

    /// Path to save the recorded audio as a WAV file. If not set, audio is not recorded to disk.
    #[clap(short, long)]
    record_to_wav: Option<String>,

    /// Disable computation and sending of partial (intermediate) transcripts.
    #[clap(long, action = clap::ArgAction::SetFalse)] // This means field defaults to true, flag makes it false
    compute_partials: bool,

    /// List available audio input devices and exit.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    list_devices: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize env_logger. Logs can be controlled by RUST_LOG environment variable.
    // e.g., RUST_LOG=whisper_stream_rs=debug,whisper_rs=info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // This is important to redirect whisper.cpp logs to the `log` facade
    install_logging_hooks();

    let args = CliArgs::parse();

    if args.list_devices {
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
                eprintln!("[Error] Could not list audio devices: {}", e);
            }
        }
        return Ok(());
    }

    let mut params = TranscriptionStreamParams::default();

    if let Some(lang) = args.language {
        params.language = Some(lang);
    }
    if let Some(step) = args.step_ms {
        params.step_ms = step;
    }
    if let Some(len) = args.length_ms {
        params.length_ms = len;
    }
    if let Some(keep) = args.keep_ms {
        params.keep_ms = keep;
    }
    if let Some(tokens) = args.max_tokens {
        params.max_tokens = tokens;
    }
    if let Some(threads) = args.n_threads {
        params.n_threads = threads;
    }
    if let Some(device_name) = args.audio_device_name {
        params.audio_device_name = Some(device_name);
    }
    if let Some(record_path) = args.record_to_wav {
        params.record_to_wav = Some(record_path);
    }
    // args.compute_partials directly sets the value due to `action = clap::ArgAction::SetFalse`
    // The field `compute_partials` in `CliArgs` must be initialized to `true` for this to work as "disable"
    // Or, more simply, if `compute_partials` in `CliArgs` defaults to `true` (its initial value before parsing),
    // and `SetFalse` makes it false if the flag is present.
    // Let's adjust CliArgs.compute_partials to default true to match TranscriptionStreamParams behavior.
    // Actually, current clap setup: `compute_partials: bool` will be true by default (if not specified in CLI) because its
    // type is `bool` and `default_value_t` is not used for it in clap derive. The action `SetFalse` will set it to false
    // if `--disable-partials` is used. This means `params.compute_partials = args.compute_partials;` is correct.
    // Hold on, for `action = clap::ArgAction::SetFalse`, the field should represent the final desired state.
    // If the field in `CliArgs` is `compute_partials`, and it has `action = clap::ArgAction::SetFalse`,
    // it means this field will be `false` IF the flag is present, and `true` OTHERWISE (if no default is specified for the field itself).
    // This aligns well if `TranscriptionStreamParams::default().compute_partials` is true.
    // Let's confirm TranscriptionStreamParams default. It is `true`. So this is correct.
    params.compute_partials = args.compute_partials;

    println!("--- Transcription Configuration ---");
    println!("Audio Device:     {}", params.audio_device_name.as_deref().unwrap_or("Default System Device"));
    println!("Step Duration:    {}ms", params.step_ms);
    println!("Window Length:    {}ms", params.length_ms);
    println!("Keep Context:     {}ms", params.keep_ms);
    println!("Max Tokens:       {}", params.max_tokens);
    println!("Threads:          {}", params.n_threads);
    println!("Language:         {}", params.language.as_deref().unwrap_or("auto"));
    println!("Compute Partials: {}", params.compute_partials);
    println!("Record to WAV:    {}", params.record_to_wav.as_deref().unwrap_or("Disabled"));
    println!("---------------------------------");

    let rx = start_transcription_stream(params);

    println!("\n[System] Start speaking... (Press Ctrl+C to stop)\n");

    let mut partial_counter: u32 = 0;
    // No longer need last_printed_partial if segments always start with \r and end with \n indirectly

    for event in rx {
        match event {
            TranscriptionStreamEvent::ProvisionalLiveUpdate { text, is_low_quality } => {
                partial_counter += 1;
                // ProvisionalLiveUpdates overwrite the current line
                print!("\r[P{}] (Low Quality: {}) {}\x1b[K", partial_counter, is_low_quality, text);
                let _ = stdout().flush();
            }
            TranscriptionStreamEvent::SegmentTranscript { text, is_low_quality } => {
                // SegmentTranscripts also overwrite the current line (where provisionals were)
                // and then we want subsequent output to be on a new line.
                // The println! handles the newline for the next distinct output.
                println!("\r[S] (Low Quality: {}) {}\x1b[K", is_low_quality, text); // Using [S] for Segment Transcript
                partial_counter = 0; // Reset counter
            }
            TranscriptionStreamEvent::SystemMessage(msg) => {
                // Ensure system messages clear the current line (if a provisional was there) and start fresh.
                println!("\r\x1b[K[System]  {}", msg);
            }
            TranscriptionStreamEvent::Error(err) => {
                // Ensure errors clear the current line and start fresh.
                let app_error = anyhow::Error::new(err).context("Transcription stream error");
                eprintln!("\r\x1b[K[Error]   {:#}", app_error);
            }
        }
    }
    Ok(())
}