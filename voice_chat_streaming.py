#!/usr/bin/env python3
"""
Voice Chat Streaming Orchestrator
Runs on Spark, connects: Client -> Whisper -> LLM (streaming) -> VibeVoice (streaming) -> Client

Architecture:
  Client sends WAV audio via WebSocket
  -> Whisper transcribes
  -> LLM streams tokens, buffered into sentences
  -> Each sentence streamed through VibeVoice
  -> PCM16 audio chunks streamed back to client

Usage:
  python voice_chat_streaming.py --port 8028

Client connects to: ws://spark:8028/voice
"""

import asyncio
import argparse
import io
import json
import re
import wave
from typing import AsyncIterator, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

# Service endpoints
WHISPER_URL = "http://localhost:8025/inference"
OLLAMA_URL = "http://localhost:11434/api/chat"
TTS_WS_URL = "ws://localhost:8027/stream"

# Configuration
SAMPLE_RATE_IN = 16000   # Input audio sample rate
SAMPLE_RATE_OUT = 24000  # VibeVoice output sample rate
LLM_MODEL = "llama3.2:3b"
SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise - 1-2 sentences max."

# Sentence boundary pattern
SENTENCE_END = re.compile(r'[.!?]\s*$')


app = FastAPI(title="Voice Chat Streaming Orchestrator")


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to Whisper and get transcription."""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        data.add_field('response_format', 'json')

        async with session.post(WHISPER_URL, data=data, timeout=30) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Whisper error: {resp.status} - {text}")
            result = await resp.json()
            return result.get("text", "").strip()


async def stream_llm_response(
    user_text: str,
    messages: list,
) -> AsyncIterator[str]:
    """Stream tokens from Ollama LLM."""
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload, timeout=60) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM error: {resp.status} - {text}")

            full_response = ""
            async for line in resp.content:
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})

            # Trim history if too long
            if len(messages) > 10:
                messages[:] = messages[:1] + messages[-8:]


async def stream_tts_audio(text: str, voice: str = None) -> AsyncIterator[bytes]:
    """Stream audio chunks from VibeVoice TTS."""
    if not text.strip():
        return

    async with aiohttp.ClientSession() as session:
        params = {"text": text}
        if voice:
            params["voice"] = voice

        url = f"{TTS_WS_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        async with session.ws_connect(url) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    yield msg.data
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    # JSON status message
                    data = json.loads(msg.data)
                    if data.get("type") == "error":
                        raise RuntimeError(f"TTS error: {data.get('message')}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    break


class SentenceBuffer:
    """Buffer LLM tokens and yield complete sentences."""

    def __init__(self):
        self.buffer = ""
        self.min_chars = 10  # Minimum chars before checking for sentence end

    def add(self, token: str) -> Optional[str]:
        """Add token, return complete sentence if available."""
        self.buffer += token

        # Check for sentence boundary
        if len(self.buffer) >= self.min_chars and SENTENCE_END.search(self.buffer):
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None

    def flush(self) -> Optional[str]:
        """Return any remaining text."""
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "voice-chat-orchestrator",
        "llm_model": LLM_MODEL,
    }


@app.websocket("/voice")
async def voice_chat(ws: WebSocket):
    """Main voice chat WebSocket endpoint.

    Protocol:
    1. Client sends: {"type": "audio", "data": "<base64 WAV>"}
    2. Server streams back:
       - {"type": "transcription", "text": "..."}
       - {"type": "llm_token", "token": "..."}
       - {"type": "audio", "data": "<base64 PCM16>"}  (or raw bytes)
       - {"type": "complete"}
    """
    await ws.accept()
    print("[orchestrator] Client connected")

    # Conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            # Wait for audio from client
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break

            if msg["type"] == "websocket.disconnect":
                break

            # Handle different message types
            if "bytes" in msg:
                # Raw audio bytes
                audio_bytes = msg["bytes"]
            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                    elif data.get("type") == "audio":
                        import base64
                        audio_bytes = base64.b64decode(data["data"])
                    else:
                        continue
                except (json.JSONDecodeError, KeyError):
                    continue
            else:
                continue

            print(f"[orchestrator] Received {len(audio_bytes)} bytes of audio")

            # Step 1: Transcribe with Whisper
            try:
                transcription = await transcribe_audio(audio_bytes)
                if not transcription:
                    await ws.send_json({"type": "error", "message": "No speech detected"})
                    continue
                print(f"[orchestrator] Transcription: {transcription}")
                await ws.send_json({"type": "transcription", "text": transcription})
            except Exception as e:
                print(f"[orchestrator] Whisper error: {e}")
                await ws.send_json({"type": "error", "message": f"Transcription failed: {e}"})
                continue

            # Step 2: Stream LLM response, buffer into sentences
            sentence_buffer = SentenceBuffer()
            full_response = ""
            sentences = []

            try:
                async for token in stream_llm_response(transcription, messages):
                    full_response += token
                    await ws.send_json({"type": "llm_token", "token": token})

                    # Check for complete sentence
                    sentence = sentence_buffer.add(token)
                    if sentence:
                        sentences.append(sentence)
                        print(f"[orchestrator] Sentence ready: {sentence[:50]}...")

                        # Stream this sentence's audio immediately
                        try:
                            async for audio_chunk in stream_tts_audio(sentence):
                                await ws.send_bytes(audio_chunk)
                        except Exception as e:
                            print(f"[orchestrator] TTS error for sentence: {e}")

                # Flush remaining text
                remaining = sentence_buffer.flush()
                if remaining:
                    sentences.append(remaining)
                    print(f"[orchestrator] Final sentence: {remaining[:50]}...")
                    try:
                        async for audio_chunk in stream_tts_audio(remaining):
                            await ws.send_bytes(audio_chunk)
                    except Exception as e:
                        print(f"[orchestrator] TTS error for final: {e}")

            except Exception as e:
                print(f"[orchestrator] LLM error: {e}")
                await ws.send_json({"type": "error", "message": f"LLM failed: {e}"})
                continue

            # Signal completion
            await ws.send_json({
                "type": "complete",
                "full_response": full_response,
                "sentences": len(sentences),
            })
            print(f"[orchestrator] Complete: {len(sentences)} sentences")

    except WebSocketDisconnect:
        print("[orchestrator] Client disconnected")
    except Exception as e:
        print(f"[orchestrator] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except:
                pass
        print("[orchestrator] Session ended")


@app.websocket("/voice-simple")
async def voice_chat_simple(ws: WebSocket):
    """Simpler voice chat - sends full response then streams all audio.

    Less complex but slightly higher latency than sentence-streaming.
    """
    await ws.accept()
    print("[orchestrator-simple] Client connected")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" not in msg:
                continue

            audio_bytes = msg["bytes"]
            print(f"[orchestrator-simple] Received {len(audio_bytes)} bytes")

            # Transcribe
            try:
                transcription = await transcribe_audio(audio_bytes)
                if not transcription:
                    continue
                await ws.send_json({"type": "transcription", "text": transcription})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue

            # Get full LLM response
            full_response = ""
            async for token in stream_llm_response(transcription, messages):
                full_response += token
                await ws.send_json({"type": "llm_token", "token": token})

            await ws.send_json({"type": "llm_complete", "text": full_response})

            # Stream TTS audio
            try:
                async for audio_chunk in stream_tts_audio(full_response):
                    await ws.send_bytes(audio_chunk)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"TTS: {e}"})

            await ws.send_json({"type": "complete"})

    except WebSocketDisconnect:
        pass
    finally:
        print("[orchestrator-simple] Session ended")


def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(description="Voice Chat Streaming Orchestrator")
    parser.add_argument("--port", type=int, default=8028)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    args = parser.parse_args()

    LLM_MODEL = args.llm_model

    print(f"Starting Voice Chat Orchestrator on {args.host}:{args.port}")
    print(f"Services: Whisper={WHISPER_URL}, LLM={OLLAMA_URL}, TTS={TTS_WS_URL}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
