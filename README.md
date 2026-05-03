
        ██╗      ██████╗  ██████╗  ██████╗ ███████╗
        ██║     ██╔═══██╗██╔════╝ ██╔═══██╗██╔════╝
        ██║     ██║   ██║██║  ███╗██║   ██║███████╗
        ██║     ██║   ██║██║   ██║██║   ██║╚════██║
        ███████╗╚██████╔╝╚██████╔╝╚██████╔╝███████║
        ╚══════╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝

                ███████╗██╗     ██╗   ██╗██╗  ██╗
                ██╔════╝██║     ██║   ██║╚██╗██╔╝
                █████╗  ██║     ██║   ██║ ╚███╔╝ 
                ██╔══╝  ██║     ██║   ██║ ██╔██╗ 
                ██║     ███████╗╚██████╔╝██╔╝ ██╗
                ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝

# Spark Voice Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Real-time voice assistant on DGX Spark with 766ms latency to first audio.

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────┐
│  Client         │     │  DGX Spark                               │
│  (mic/speakers) │ WS  │                                          │
│                 ├────►│  Whisper STT (:8025)                     │
│                 │     │       ↓                                  │
│                 │     │  Orchestrator (:8028)                    │
│                 │     │       ├──► Ollama LLM (:11434)           │
│                 │     │       │    [streams tokens]              │
│                 │     │       └──► VibeVoice TTS (:8027)         │
│                 │◄────│            [streams audio]               │
│  ◄── plays      │     │                                          │
└─────────────────┘     └──────────────────────────────────────────┘
```

## Performance

| Metric | Value |
|--------|-------|
| Time to first audio | ~766ms |
| TTS RTF | 0.48x (2x faster than real-time) |
| Total pipeline | Streaming (no waiting for full response) |

## Quick Start

### Prerequisites: Fix PyTorch CUDA on Spark

If you're seeing `CUDA available: False`, your PyTorch may not have CUDA enabled. This is a [common issue on Spark](https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/). Fix it:

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Install Dependencies

```bash
# Clone this repo
git clone https://github.com/Logos-Flux/spark-voice-pipeline.git
cd spark-voice-pipeline

# Install VibeVoice (TTS model)
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e . && cd ..

# Install Python deps (server + client)
pip install -r requirements.txt
```

> The full `requirements.txt` covers both the Spark-side services (FastAPI, uvicorn, aiohttp, scipy, torch) and the client (websockets, sounddevice, numpy). On a client-only machine you can install just `websockets sounddevice numpy`.

### Start Services (on Spark)

```bash
./start_streaming_services.sh
```

Or manually:
```bash
# Terminal 1: Whisper STT
cd whisper.cpp/build-cuda/bin
./whisper-server -m models/ggml-large-v3-turbo-q8_0.bin --host 127.0.0.1 --port 8025

# Terminal 2: VibeVoice TTS
python vibevoice_streaming_server.py  # Port 8027

# Terminal 3: Orchestrator
python voice_chat_streaming.py  # Port 8028

# Terminal 4: Ollama (if not already running)
ollama serve
```

### Run Client (on your laptop)

```bash
# Pass the Spark server's hostname or IP, or set $SPARK_HOST
python voice_chat_client_streaming.py --spark-host <spark-host-or-ip>
```

By default the services bind to `127.0.0.1` (loopback only). To use the client from another machine, either:
- Tunnel to the Spark box (SSH `-L`, Tailscale, or similar) and connect to `localhost`, **or**
- Start services with `BIND_HOST=0.0.0.0 ./start_streaming_services.sh` to expose on the LAN — read the **Security** section first; there is no built-in auth.

## Key Innovations

### 1. Sentence-Level Streaming

Instead of waiting for the full LLM response, we buffer tokens until a sentence boundary (. ! ?), then immediately stream that sentence to TTS while the LLM continues generating.

### 2. Continuous Audio Playback

Client uses `sd.OutputStream` with a callback function for gap-free audio playback, instead of discrete `sd.play()` calls which cause choppy audio.

### 3. WebSocket Throughout

Real-time bidirectional streaming at every stage eliminates HTTP request overhead.

## Available Voices

VibeVoice-Realtime-0.5B includes 7 preset voices:

| Voice | Description |
|-------|-------------|
| Emma | English female (natural, recommended) |
| Mike | English male (natural) |
| Carter | English male |
| Davis | English male |
| Frank | English male |
| Grace | English female (older sounding) |
| Samuel | Indian English male |

Note: The 0.5B model doesn't support voice cloning. For custom voices, use the 1.5B model.

## Files

```
vibevoice_streaming_server.py    # TTS server (port 8027)
voice_chat_streaming.py          # Orchestrator (port 8028)
voice_chat_client_streaming.py   # Client with continuous playback
start_streaming_services.sh      # Startup script
```

## Hardware

Tested on:
- **Server:** DGX Spark (GB10 GPU, CUDA 13, 128GB unified memory)
- **Client:** Windows laptop with mic/speakers

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_HOST` | `localhost` | Client target — Spark hostname or IP |
| `BIND_HOST` | `127.0.0.1` | `start_streaming_services.sh` server bind address |
| `WHISPER_URL` | `http://localhost:8025/inference` | Orchestrator → Whisper endpoint |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Orchestrator → Ollama endpoint |
| `TTS_WS_URL` | `ws://localhost:8027/stream` | Orchestrator → TTS WebSocket |

## Security

The three services (Whisper, VibeVoice, Orchestrator) ship with **no authentication**. Defaults bind to `127.0.0.1` so a fresh install is loopback-only.

If you change the bind address to `0.0.0.0` (or any non-loopback interface), **anyone on the network can:**
- Drive the GPU via the TTS / orchestrator endpoints (free compute)
- Send arbitrary prompts to your local Ollama via the voice pipeline
- Open WebSocket connections from any browser tab the operator visits (no Origin check)

Recommended deployment patterns when remote access is needed:
- **SSH local forward** — `ssh -L 8028:localhost:8028 spark-host`, then connect the client to `localhost`.
- **Tailscale** (or another auth'd overlay) — bind to the tailnet interface; tailnet membership becomes auth.
- **Reverse proxy with auth** — front the services with Caddy/nginx/Traefik enforcing a bearer token or mTLS.

Do not expose these ports directly to the public internet.

Other notes:
- `start_streaming_services.sh` writes server logs to `~/ggml-org/logs/*.log`. These contain transcribed user speech and assistant replies and are not rotated — clear or rotate them yourself if that matters for your use case.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR guidelines. For security issues, see [SECURITY.md](SECURITY.md). Release notes live in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)

## Credits

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [Ollama](https://ollama.ai)

