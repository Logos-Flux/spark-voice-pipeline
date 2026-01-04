#!/usr/bin/env python3
"""
Voice Chat Streaming Client - Smooth continuous audio playback.

Usage:
    python voice_chat_client_streaming.py --spark-host 10.0.0.104

Requirements:
    pip install sounddevice numpy websockets
"""

import argparse
import asyncio
import io
import json
import queue
import sys
import threading
import time
import wave

import numpy as np

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("Install: pip install sounddevice")
    sys.exit(1)

# Audio settings
SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHUNK_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE_IN * CHUNK_MS / 1000)

# VAD
SILENCE_THRESHOLD = 50
SILENCE_DURATION = 1.0
MIN_SPEECH = 0.3
MAX_RECORD = 10.0


class StreamingPlayer:
    """Continuous audio playback using output stream callback."""

    def __init__(self):
        self.buffer = b""
        self.lock = threading.Lock()
        self.stream = None
        self.done = False

    def start(self):
        self.buffer = b""
        self.done = False
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE_OUT,
            channels=1,
            dtype=np.int16,
            callback=self._callback,
            blocksize=1024,
        )
        self.stream.start()

    def add(self, pcm_bytes: bytes):
        with self.lock:
            self.buffer += pcm_bytes

    def finish(self):
        """Signal no more data coming."""
        self.done = True

    def stop(self):
        """Stop and wait for buffer to drain."""
        self.done = True
        # Wait for buffer to empty
        for _ in range(100):  # Max 10 seconds
            with self.lock:
                if len(self.buffer) == 0:
                    break
            time.sleep(0.1)

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        bytes_needed = frames * 2  # 16-bit = 2 bytes per sample

        with self.lock:
            if len(self.buffer) >= bytes_needed:
                data = self.buffer[:bytes_needed]
                self.buffer = self.buffer[bytes_needed:]
            elif len(self.buffer) > 0:
                # Pad with what we have
                data = self.buffer + b'\x00' * (bytes_needed - len(self.buffer))
                self.buffer = b""
            else:
                # No data - output silence
                data = b'\x00' * bytes_needed

        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)


class VoiceClient:
    def __init__(self, host: str, port: int = 8028):
        self.ws_url = f"ws://{host}:{port}/voice"
        self.player = StreamingPlayer()

    def draw_level(self, energy: float, is_speech: bool, speaking: int):
        level = min(20, int(energy / 25))
        bar = "#" * level + "-" * (20 - level)
        if speaking > 0:
            status = f"REC ({speaking * CHUNK_MS / 1000:.1f}s)"
        elif is_speech:
            status = "Voice"
        else:
            status = "..."
        print(f"\r [{bar}] {status: <20}", end="", flush=True)

    def record(self) -> bytes:
        chunks = []
        silent = 0
        speaking = 0
        max_silent = int(SILENCE_DURATION * 1000 / CHUNK_MS)
        min_speak = int(MIN_SPEECH * 1000 / CHUNK_MS)
        max_chunks = int(MAX_RECORD * 1000 / CHUNK_MS)

        print("Speak now...")

        with sd.InputStream(samplerate=SAMPLE_RATE_IN, channels=1,
                           dtype=np.int16, blocksize=CHUNK_SIZE) as mic:
            while len(chunks) < max_chunks:
                data, _ = mic.read(CHUNK_SIZE)
                data = data.flatten()
                energy = np.sqrt(np.mean(data.astype(float)**2))
                is_speech = energy > SILENCE_THRESHOLD

                if is_speech:
                    speaking += 1
                    silent = 0
                    chunks.append(data)
                else:
                    silent += 1
                    if speaking > 0:
                        chunks.append(data)
                    if silent >= max_silent and speaking >= min_speak:
                        break
                self.draw_level(energy, is_speech, speaking)

        print("\r" + " " * 50 + "\r", end="")

        if not chunks:
            return b""

        audio = np.concatenate(chunks)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE_IN)
            w.writeframes(audio.tobytes())
        buf.seek(0)
        print(f"Recorded {len(audio) / SAMPLE_RATE_IN:.1f}s")
        return buf.read()

    async def chat_turn(self, audio_bytes: bytes) -> bool:
        try:
            async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                t_start = time.time()
                await ws.send(audio_bytes)

                response_text = ""
                first_audio = None
                audio_chunks = 0

                # Start continuous playback stream
                self.player.start()

                async for msg in ws:
                    if isinstance(msg, bytes):
                        if first_audio is None:
                            first_audio = time.time() - t_start
                            print(f"\n[First audio: {first_audio*1000:.0f}ms]")
                        self.player.add(msg)
                        audio_chunks += 1

                    elif isinstance(msg, str):
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "transcription":
                            t_stt = time.time() - t_start
                            print(f"You: {data.get('text', '')}  ({t_stt:.1f}s)")

                        elif msg_type == "llm_token":
                            token = data.get("token", "")
                            response_text += token
                            print(token, end="", flush=True)

                        elif msg_type == "complete":
                            total = time.time() - t_start
                            print(f"\n[{audio_chunks} chunks, {total:.1f}s total]")
                            break

                        elif msg_type == "error":
                            print(f"\nError: {data.get('message')}")
                            break

                # Wait for playback to complete
                self.player.finish()
                self.player.stop()
                return True

        except Exception as e:
            print(f"\nError: {e}")
            self.player.stop()
            return False

    async def run(self):
        print("\n" + "="*45)
        print("  Streaming Voice Chat (Low Latency)")
        print("  Ctrl+C to exit")
        print("="*45 + "\n")

        while True:
            try:
                audio_bytes = self.record()
                if len(audio_bytes) < 1000:
                    continue
                await self.chat_turn(audio_bytes)
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spark-host", default="10.0.0.104")
    parser.add_argument("--port", type=int, default=8028)
    args = parser.parse_args()

    print(f"\n  Streaming Voice Chat -> {args.spark_host}\n")
    client = VoiceClient(args.spark_host, args.port)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
