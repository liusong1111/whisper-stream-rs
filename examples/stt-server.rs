use axum::{
    Json, Router,
    extract::State,
    response::{IntoResponse, Sse, sse::Event as SseEvent},
    routing::post,
};
use clap::Parser;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::sync::broadcast;
use tokio::sync::broadcast::{Receiver as BroadcastReceiver, Sender as BroadcastSender};
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};
use tower_http::cors::CorsLayer;

use whisper_stream_rs::{
    DEFAULT_MODEL, Event, WhisperStream, WhisperStreamBuilder, WhisperStreamError,
};

#[derive(Clone, Default, Serialize, Deserialize)]
struct WhisperEvent {
    // ProvisionalLiveUpdate, SegmentTranscript,SystemMessage, Error
    // ==>
    // live, segment, system, error
    pub event_type: String,
    pub text: String,
    pub is_low_quality: bool,
}

impl From<whisper_stream_rs::Event> for WhisperEvent {
    fn from(event: whisper_stream_rs::Event) -> Self {
        match event {
            Event::ProvisionalLiveUpdate {
                text,
                is_low_quality,
            } => Self {
                event_type: "live".to_string(),
                text,
                is_low_quality,
            },
            Event::SegmentTranscript {
                text,
                is_low_quality,
            } => Self {
                event_type: "segment".to_string(),
                text,
                is_low_quality,
            },
            Event::SystemMessage(text) => Self {
                event_type: "system".to_string(),
                text,
                is_low_quality: false,
            },
            Event::Error(error) => Self {
                event_type: "error".to_string(),
                text: error.to_string(),
                is_low_quality: false,
            },
        }
    }
}
// impl Into<SseEvent> for WhisperEvent {}

impl WhisperClient {
    pub fn to_sse(self) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>> + Send> {
        let stream = BroadcastStream::new(self.rx);
        Sse::new(stream.map(|result| match result {
            Ok(event) => Ok(SseEvent::default().json_data(event).unwrap()),
            Err(BroadcastStreamRecvError::Lagged(_)) => {
                Ok(SseEvent::default().event("lagged").data(""))
            }
        }))
    }
}

#[derive(Clone)]
struct AppState {
    // whisper_instance: Arc<Mutex<Option<WhisperInstance>>>,
    builder: WhisperStreamBuilder,
    tx: BroadcastSender<WhisperEvent>,
}

struct WhisperClient {
    rx: BroadcastReceiver<WhisperEvent>,
}

impl AppState {
    pub fn new(builder: WhisperStreamBuilder) -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            // whisper_instance: Arc::new(Mutex::new(None)),
            builder,
            tx,
        }
    }
    // 系统启动时就启动
    pub fn start(&self) -> Result<(), WhisperStreamError> {
        // let mut instance = self.whisper_instance.lock().await;
        let _instance = self.builder.clone().build()?;
        let rx = _instance.rx;
        let tx = self.tx.clone();
        tokio::task::spawn_blocking(move || {
            while let Ok(event) = rx.recv() {
                let _ = tx.send(event.into());
            }
        });
        // let _ = instance.insert(_instance);
        Ok(())
    }

    pub async fn subscribe(&self) -> WhisperClient {
        let rx = self.tx.subscribe();
        WhisperClient { rx }
    }

    // pub async fn is_running(&self) -> bool {
    //     let instance = self.whisper_instance.lock().await;
    //     instance.is_some()
    // }
    // pub async fn status(&self) -> String {
    //     match self.is_running().await {
    //         true => "running".to_string(),
    //         false => "stopped".to_string(),
    //     }
    // }
    // pub async fn start(&self) -> Result<(), WhisperStreamError> {
    //     if self.is_running().await {
    //         return Ok(());
    //     }
    //     let mut instance = self.whisper_instance.lock().await;
    //     let _instance = self.builder.clone().build()?;
    //     let _ = instance.insert(_instance);
    //     Ok(())
    // }
    // pub async fn stop(&self) -> Result<(), WhisperStreamError> {
    //     if !self.is_running().await {
    //         return Ok(());
    //     }
    //     let mut instance = self.whisper_instance.lock().await;
    //     let _ = instance.take();
    //     Ok(())
    // }
}

async fn start_api_server(port: i16, _args: CliArgs, builder: WhisperStreamBuilder) {
    let cors = CorsLayer::very_permissive();
    let state = AppState::new(builder);
    state.start().expect("Failed to start WhisperStream");

    let app: Router = Router::new()
        // .route("/api/stt_start", get(stt_start_handler))
        // .route("/api/stt_stop", get(stt_stop_handler))
        // .route("/api/stt_status", get(stt_status_handler))
        .route("/api/stt_events", post(stt_events_handler))
        .layer(cors)
        .with_state(state);

    let addr = format!("127.0.0.1:{port}");
    let tcp_listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("listening on {}", addr);
    axum::serve(tcp_listener, app)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.unwrap() })
        .await
        .unwrap();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttEventsRequest {}

// async fn stt_start_handler(State(state): State<AppState>) -> impl IntoResponse {
//     match state.start().await {
//         Ok(_) => Json(json!({ "status": "ok" })),
//         Err(err) => Json(json!({ "status": "error", "error": err.to_string() })),
//     }
// }

// async fn stt_stop_handler(State(state): State<AppState>) -> impl IntoResponse {
//     match state.stop().await {
//         Ok(_) => Json(json!({ "status": "ok" })),
//         Err(err) => Json(json!({ "status": "error", "error": err.to_string() })),
//     }
// }

// async fn stt_status_handler(State(state): State<AppState>) -> impl IntoResponse {
//     let status = state.status().await;
//     Json(json!({ "status": status }))
// }

async fn stt_events_handler(
    State(state): State<AppState>,
    Json(_payload): Json<SttEventsRequest>,
) -> impl IntoResponse {
    let cli = state.subscribe().await;
    cli.to_sse()
    // Json(json!({ "status": "ok" }))
}

/// Command-line tool to stream audio from a microphone and transcribe it using whisper-stream-rs.
#[derive(Parser, Debug, Clone)]
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

    #[clap(long, default_value_t = 8072)]
    port: i16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize env_logger. Logs can be controlled by RUST_LOG environment variable.
    // e.g., RUST_LOG=whisper_stream_rs=debug,whisper_rs=info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = CliArgs::parse();
    let args_clone = args.clone();

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
    // return builder;
    //
    start_api_server(args.port, args_clone, builder).await;
    Ok(())
}

// fn build_whisper_instance(_args: CliArgs, builder: WhisperStreamBuilder) -> anyhow::Result<()> {
//     let WhisperInstance { stream: _, rx } = builder.build()?;

//     println!("\n[System] Start speaking... (Press Ctrl+C to stop)\n");

//     let mut partial_counter: u32 = 0;
//     // No longer need last_printed_partial if segments always start with \r and end with \n indirectly

//     for event in rx {
//         match event {
//             Event::ProvisionalLiveUpdate {
//                 text,
//                 is_low_quality,
//             } => {
//                 partial_counter += 1;
//                 // ProvisionalLiveUpdates overwrite the current line
//                 print!(
//                     "\r[P{}] (Low Quality: {}) {}\x1b[K",
//                     partial_counter, is_low_quality, text
//                 );
//                 let _ = stdout().flush();
//             }
//             Event::SegmentTranscript {
//                 text,
//                 is_low_quality,
//             } => {
//                 // SegmentTranscripts also overwrite the current line (where provisionals were)
//                 // and then we want subsequent output to be on a new line.
//                 // The println! handles the newline for the next distinct output.
//                 println!("\r[S] (Low Quality: {}) {}\x1b[K", is_low_quality, text); // Using [S] for Segment Transcript
//                 partial_counter = 0; // Reset counter
//             }
//             Event::SystemMessage(msg) => {
//                 // Ensure system messages clear the current line (if a provisional was there) and start fresh.
//                 println!("\r\x1b[K[System]  {}", msg);
//             }
//             Event::Error(err) => {
//                 // Ensure errors clear the current line and start fresh.
//                 let app_error = anyhow::Error::new(err).context("Transcription stream error");
//                 eprintln!("\r\x1b[K[Error]   {:#}", app_error);
//             }
//         }
//     }
//     Ok(())
// }
