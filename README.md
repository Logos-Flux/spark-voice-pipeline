
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

Real-time voice assistant on DGX Spark with 766ms latency to first audio.

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────┐
│  Client         │     │  DGX Spark (10.0.0.104)                  │
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

# Install VibeVoice
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e . && cd ..

# Install Python deps
pip install websockets sounddevice numpy
```

### Start Services (on Spark)

```bash
./start_streaming_services.sh
```

Or manually:
```bash
# Terminal 1: Whisper STT
cd whisper.cpp/build-cuda/bin
./whisper-server -m models/ggml-large-v3-turbo-q8_0.bin --host 0.0.0.0 --port 8025

# Terminal 2: VibeVoice TTS
python vibevoice_streaming_server.py  # Port 8027

# Terminal 3: Orchestrator
python voice_chat_streaming.py  # Port 8028

# Terminal 4: Ollama (if not already running)
ollama serve
```

### Run Client (on your laptop)

```bash
python voice_chat_client_streaming.py --spark-host 10.0.0.104
```

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

## License

MIT

## Credits

- [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [Ollama](https://ollama.ai)
```

