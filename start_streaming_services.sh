#!/bin/bash
# Start Streaming Voice Chat Services on DGX Spark
# ================================================
#
# Usage:
#   ./start_streaming_services.sh

set -e

echo "=========================================="
echo "  Starting Streaming Voice Chat Services"
echo "=========================================="

mkdir -p ~/ggml-org/logs

# Kill any existing instances
echo "Stopping existing services..."
pkill -f whisper-server 2>/dev/null || true
pkill -f vibevoice_streaming_server 2>/dev/null || true
pkill -f voice_chat_streaming 2>/dev/null || true
sleep 2

# Start Whisper Server
echo ""
echo "[1/3] Starting Whisper STT server (port 8025)..."
cd ~/ggml-org/whisper.cpp/build-cuda/bin
LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH nohup ./whisper-server \
    -m ~/ggml-org/whisper.cpp/models/ggml-large-v3-turbo-q8_0.bin \
    --host 0.0.0.0 \
    --port 8025 \
    > ~/ggml-org/logs/whisper-server.log 2>&1 &
echo "      PID: $!"

# Start VibeVoice Streaming TTS
echo ""
echo "[2/3] Starting VibeVoice Streaming TTS (port 8027)..."
echo "      This takes ~30s to load the model..."
cd ~/ggml-org
nohup python3 vibevoice_streaming_server.py --port 8027 --voice en-Emma_woman \
    > logs/vibevoice-streaming.log 2>&1 &
echo "      PID: $!"

# Wait for services
echo ""
echo "Waiting for services to be ready..."

# Wait for Whisper
for i in {1..30}; do
    if curl -s http://localhost:8025/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "  [OK] Whisper STT"
        break
    fi
    sleep 1
done

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  [OK] Ollama LLM"
else
    echo "  [!] Ollama not running - start with: ollama serve"
fi

# Wait for VibeVoice (longer)
echo "  [..] Waiting for VibeVoice (~30s)..."
for i in {1..60}; do
    if curl -s http://localhost:8027/health 2>/dev/null | grep -q '"status":"ok"'; then
        echo "  [OK] VibeVoice Streaming TTS"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  [!] VibeVoice not ready - check logs/vibevoice-streaming.log"
    fi
    sleep 1
done

# Start Orchestrator
echo ""
echo "[3/3] Starting Voice Chat Orchestrator (port 8028)..."
cd ~/ggml-org
nohup python3 voice_chat_streaming.py --port 8028 \
    > logs/orchestrator.log 2>&1 &
echo "      PID: $!"

sleep 2
if curl -s http://localhost:8028/health 2>/dev/null | grep -q '"status":"ok"'; then
    echo "  [OK] Orchestrator"
else
    echo "  [!] Orchestrator may still be starting..."
fi

echo ""
echo "=========================================="
echo "  Streaming Voice Chat Services Ready"
echo "=========================================="
echo ""
echo "Services:"
echo "  Whisper STT:      http://localhost:8025"
echo "  Ollama LLM:       http://localhost:11434"
echo "  VibeVoice TTS:    ws://localhost:8027/stream"
echo "  Orchestrator:     ws://localhost:8028/voice"
echo ""
echo "Client usage (run on your laptop):"
echo "  python voice_chat_client_streaming.py --spark-host 10.0.0.104"
echo ""
echo "Expected latency: ~800ms to first audio"
echo ""
echo "Logs: ~/ggml-org/logs/"
echo "=========================================="
