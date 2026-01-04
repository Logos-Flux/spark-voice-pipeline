#!/usr/bin/env python3
"""
VibeVoice Streaming TTS Server
WebSocket-based streaming for lowest latency voice synthesis.

Endpoints:
  WS  /stream?text=...     - Stream audio for text (query param)
  WS  /stream              - Stream audio for text (send JSON messages)
  GET /health              - Health check
  GET /voices              - List available voices

Usage:
  python vibevoice_streaming_server.py --port 8027 --voice en-Emma_woman
"""

import asyncio
import argparse
import copy
import json
import os
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

# Constants
MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
VOICES_DIR = Path(__file__).parent / "VibeVoice/demo/voices/streaming_model"
SAMPLE_RATE = 24000
DEFAULT_PORT = 8027


class StreamingTTSService:
    """Streaming TTS service using VibeVoice-Realtime."""

    def __init__(self, model_path: str, device: str = "cuda", inference_steps: int = 5):
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE
        self.device = device
        self._torch_device = torch.device(device)

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Any] = {}

    def load(self) -> None:
        """Load model and processor."""
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Device-specific settings
        if self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = self.device
            attn_impl = "sdpa"

        print(f"[startup] Loading model with dtype={load_dtype}, attn={attn_impl}")

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                print(f"[startup] flash_attention_2 failed, using SDPA: {e}")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        # Load voice presets
        self.voice_presets = self._load_voice_presets()
        print(f"[startup] Found {len(self.voice_presets)} voice presets")

    def _load_voice_presets(self) -> Dict[str, Path]:
        """Scan for available voice presets."""
        if not VOICES_DIR.exists():
            raise RuntimeError(f"Voices directory not found: {VOICES_DIR}")

        presets = {}
        for pt_path in VOICES_DIR.glob("*.pt"):
            presets[pt_path.stem] = pt_path
        return dict(sorted(presets.items()))

    def load_voice(self, voice_key: str) -> None:
        """Load a voice preset into cache."""
        if voice_key not in self.voice_presets:
            raise ValueError(f"Voice not found: {voice_key}")

        if voice_key not in self._voice_cache:
            path = self.voice_presets[voice_key]
            print(f"[voice] Loading {voice_key} from {path}")
            self._voice_cache[voice_key] = torch.load(
                path, map_location=self._torch_device, weights_only=False
            )
        self.default_voice_key = voice_key

    def _get_voice(self, voice_key: Optional[str] = None) -> Any:
        """Get voice preset data."""
        key = voice_key or self.default_voice_key
        if key not in self._voice_cache:
            self.load_voice(key)
        return self._voice_cache[key]

    def _prepare_inputs(self, text: str, voice_data: Any) -> Dict:
        """Prepare model inputs."""
        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip().replace("'", "'"),
            cached_prompt=voice_data,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {k: v.to(self._torch_device) if hasattr(v, "to") else v
                for k, v in processed.items()}

    def stream(
        self,
        text: str,
        voice_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """Stream audio chunks for given text."""
        if not text.strip():
            return

        voice_data = self._get_voice(voice_key)
        inputs = self._prepare_inputs(text, voice_data)

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors = []
        stop_signal = stop_event or threading.Event()

        def run_generation():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_signal.is_set,
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(voice_data),
                )
            except Exception as e:
                errors.append(e)
                traceback.print_exc()
                audio_streamer.end()

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        try:
            stream = audio_streamer.get_stream(0)
            for chunk in stream:
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)

                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)

                # Normalize
                peak = np.max(np.abs(chunk)) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak

                yield chunk.astype(np.float32)
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join(timeout=5.0)
            if errors:
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        """Convert float32 audio chunk to PCM16 bytes."""
        chunk = np.clip(chunk, -1.0, 1.0)
        return (chunk * 32767.0).astype(np.int16).tobytes()


# FastAPI app
app = FastAPI(title="VibeVoice Streaming TTS")
service: Optional[StreamingTTSService] = None


@app.on_event("startup")
async def startup():
    global service
    model_path = os.environ.get("MODEL_PATH", MODEL_PATH)
    device = os.environ.get("DEVICE", "cuda")
    voice = os.environ.get("VOICE", "en-Emma_woman")

    service = StreamingTTSService(model_path=model_path, device=device)
    service.load()
    service.load_voice(voice)

    app.state.service = service
    app.state.ws_lock = asyncio.Lock()
    print(f"[startup] Ready with voice: {voice}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "vibevoice-realtime-0.5b",
        "streaming": True,
        "sample_rate": SAMPLE_RATE,
        "voice": service.default_voice_key if service else None,
    }


@app.get("/voices")
async def voices():
    if not service:
        return {"voices": []}
    return {
        "voices": list(service.voice_presets.keys()),
        "default": service.default_voice_key,
    }


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    """WebSocket endpoint for streaming TTS.

    Connect with text as query param: /stream?text=Hello
    Or send JSON messages: {"text": "Hello", "voice": "en-Emma_woman"}

    Receives: PCM16 audio bytes at 24kHz
    """
    await ws.accept()

    # Check if service is busy
    lock = app.state.ws_lock
    if lock.locked():
        await ws.send_json({"type": "error", "message": "Service busy"})
        await ws.close(code=1013)
        return

    async with lock:
        # Get text from query params or wait for message
        text = ws.query_params.get("text", "")
        voice = ws.query_params.get("voice")
        cfg = float(ws.query_params.get("cfg", "1.5"))

        if not text:
            # Wait for JSON message with text
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                text = msg.get("text", "")
                voice = msg.get("voice", voice)
                cfg = msg.get("cfg", cfg)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "error", "message": "Timeout waiting for text"})
                await ws.close()
                return
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                await ws.close()
                return

        if not text:
            await ws.send_json({"type": "error", "message": "No text provided"})
            await ws.close()
            return

        print(f"[stream] Starting: {text[:50]}...")
        await ws.send_json({"type": "start", "text_length": len(text)})

        stop_event = threading.Event()
        chunk_count = 0
        total_samples = 0

        try:
            iterator = service.stream(text, voice_key=voice, cfg_scale=cfg, stop_event=stop_event)
            sentinel = object()

            while ws.client_state == WebSocketState.CONNECTED:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break

                pcm_bytes = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(pcm_bytes)

                chunk_count += 1
                total_samples += len(chunk)

                if chunk_count == 1:
                    await ws.send_json({"type": "first_audio"})

        except WebSocketDisconnect:
            print("[stream] Client disconnected")
            stop_event.set()
        except Exception as e:
            print(f"[stream] Error: {e}")
            traceback.print_exc()
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except:
                pass
        finally:
            stop_event.set()
            duration = total_samples / SAMPLE_RATE if total_samples else 0
            print(f"[stream] Complete: {chunk_count} chunks, {duration:.2f}s audio")

            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_json({
                        "type": "complete",
                        "chunks": chunk_count,
                        "duration": duration,
                    })
                    await ws.close()
                except:
                    pass


@app.post("/synthesize")
async def synthesize_batch(request: dict):
    """Non-streaming batch synthesis (for compatibility)."""
    text = request.get("text", "")
    voice = request.get("voice")

    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)

    # Collect all chunks
    chunks = []
    for chunk in service.stream(text, voice_key=voice):
        chunks.append(chunk)

    if not chunks:
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Combine and convert to WAV
    import io
    import scipy.io.wavfile as wav

    audio = np.concatenate(chunks)
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, audio_int16)
    buf.seek(0)

    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="audio/wav")


def main():
    parser = argparse.ArgumentParser(description="VibeVoice Streaming TTS Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--voice", default="en-Emma_woman")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = MODEL_PATH
    os.environ["DEVICE"] = args.device
    os.environ["VOICE"] = args.voice

    print(f"Starting VibeVoice Streaming TTS on {args.host}:{args.port}")
    print(f"Voice: {args.voice}, Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
