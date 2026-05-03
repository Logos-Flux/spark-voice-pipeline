# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-03

First public release.

### Added

- Real-time voice assistant pipeline on DGX Spark: Whisper STT → Ollama
  LLM (streaming) → VibeVoice TTS (streaming) → client.
- Sentence-level streaming: orchestrator buffers LLM tokens until a
  sentence boundary, then streams that sentence to TTS while the LLM
  continues generating. ~766ms time to first audio.
- Continuous client-side audio playback via `sounddevice` callback to
  avoid gaps between chunks.
- WebSocket transport at every stage to eliminate HTTP overhead.
- Configurable service endpoints via `WHISPER_URL`, `OLLAMA_URL`,
  `TTS_WS_URL` env vars; client target via `SPARK_HOST`; service bind
  address via `BIND_HOST`.
- Security section in README documenting the lack of built-in auth and
  recommended deployment patterns (SSH `-L`, Tailscale, reverse proxy).

### Defaults

- All three services bind `127.0.0.1` by default. LAN exposure is
  opt-in via `--host 0.0.0.0` or `BIND_HOST=0.0.0.0`.

[1.0.0]: https://github.com/Logos-Flux/spark-voice-pipeline/releases/tag/v1.0.0
