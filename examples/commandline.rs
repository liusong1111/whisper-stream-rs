use clap::Parser;
use std::io::{Write, stdout};
use whisper_stream_rs::{DEFAULT_MODEL, Event, WhisperInstance, WhisperStream};

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

    /// Model to use for transcription (e.g., "base.en", "tiny.en", "small.en").
    #[clap(long)]
    model: Option<String>,

    /// Disable computation and sending of partial (intermediate) transcripts.
    #[clap(long, action = clap::ArgAction::SetFalse)]
    compute_partials: bool,

    /// List available audio input devices and exit.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    list_devices: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize env_logger. Logs can be controlled by RUST_LOG environment variable.
    // e.g., RUST_LOG=whisper_stream_rs=debug,whisper_rs=info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = CliArgs::parse();

    if args.list_devices {
        println!("Available audio input devices:");
        match WhisperStream::list_devices() {
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

    let audio_device_name = args.audio_device_name.clone();
    let language = args.language.clone();
    let record_to_wav = args.record_to_wav.clone();
    let mut builder = WhisperStream::builder();

    if let Some(lang) = args.language {
        builder = builder.language(&lang);
    }
    if let Some(step) = args.step_ms {
        builder = builder.step_ms(step);
    }
    if let Some(len) = args.length_ms {
        builder = builder.length_ms(len);
    }
    if let Some(keep) = args.keep_ms {
        builder = builder.keep_ms(keep);
    }
    if let Some(tokens) = args.max_tokens {
        builder = builder.max_tokens(tokens);
    }
    if let Some(threads) = args.n_threads {
        if threads > 0 {
            builder = builder.n_threads(threads);
        } else {
            anyhow::bail!("--n-threads must be a positive integer (got {threads})");
        }
    }
    if let Some(device_name) = args.audio_device_name {
        builder = builder.device(&device_name);
    }
    if let Some(record_path) = args.record_to_wav {
        builder = builder.record_to_wav(&record_path);
    }
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    builder = builder.model(model.clone());

    builder = builder.compute_partials(args.compute_partials);

    println!("--- Transcription Configuration ---");
    println!(
        "Audio Device:     {}",
        audio_device_name
            .as_deref()
            .unwrap_or("Default System Device")
    );
    println!("Model:            {}", model);
    println!("Step Duration:    {}ms", args.step_ms.unwrap_or(800));
    println!("Window Length:    {}ms", args.length_ms.unwrap_or(5000));
    println!("Keep Context:     {}ms", args.keep_ms.unwrap_or(200));
    println!("Max Tokens:       {}", args.max_tokens.unwrap_or(32));
    println!(
        "Threads:          {}",
        args.n_threads
            .unwrap_or_else(|| std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4))
    );
    println!(
        "Language:         {}",
        language.as_deref().unwrap_or("auto")
    );
    println!("Compute Partials: {}", args.compute_partials);
    println!(
        "Record to WAV:    {}",
        record_to_wav.as_deref().unwrap_or("Disabled")
    );
    println!("---------------------------------");

    let WhisperInstance { stream: _, rx } = builder.build()?;

    println!("\n[System] Start speaking... (Press Ctrl+C to stop)\n");

    let mut partial_counter: u32 = 0;
    // No longer need last_printed_partial if segments always start with \r and end with \n indirectly

    for event in rx {
        match event {
            Event::ProvisionalLiveUpdate {
                text,
                is_low_quality,
            } => {
                partial_counter += 1;
                // ProvisionalLiveUpdates overwrite the current line
                print!(
                    "\r[P{}] (Low Quality: {}) {}\x1b[K",
                    partial_counter, is_low_quality, text
                );
                let _ = stdout().flush();
            }
            Event::SegmentTranscript {
                text,
                is_low_quality,
            } => {
                // SegmentTranscripts also overwrite the current line (where provisionals were)
                // and then we want subsequent output to be on a new line.
                // The println! handles the newline for the next distinct output.
                println!("\r[S] (Low Quality: {}) {}\x1b[K", is_low_quality, text); // Using [S] for Segment Transcript
                partial_counter = 0; // Reset counter
            }
            Event::SystemMessage(msg) => {
                // Ensure system messages clear the current line (if a provisional was there) and start fresh.
                println!("\r\x1b[K[System]  {}", msg);
            }
            Event::Error(err) => {
                // Ensure errors clear the current line and start fresh.
                let app_error = anyhow::Error::new(err).context("Transcription stream error");
                eprintln!("\r\x1b[K[Error]   {:#}", app_error);
            }
        }
    }
    Ok(())
}
