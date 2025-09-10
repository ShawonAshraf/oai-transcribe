from dotenv import load_dotenv
import asyncio
import struct
import base64
import json
import os
import time
import logging
from typing import List
from pathlib import Path

import nest_asyncio
import numpy as np
from openai import OpenAI
import resampy
import soundfile as sf
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-transcribe"
TARGET_SR = 24_000
PCM_SCALE = 32_767
CHUNK_SAMPLES = 3_072
RT_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_DONE = "conversation.item.input_audio_transcription.completed"

app = FastAPI(title="Real-time Audio Transcription Service")

# ── Original helper functions from main.py ────────────────────────────────────
def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

def load_and_resample(path: str, sr: int = TARGET_SR) -> np.ndarray:
    """Return mono PCM-16 as a NumPy array."""
    data, file_sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = resampy.resample(data, file_sr, sr)
    return data

async def _send_audio(ws, pcm: np.ndarray, chunk: int, sr: int) -> None:
    """Producer: stream base-64 chunks at real-time pace, then signal EOF."""
    dur = 0.025
    t_next = time.monotonic()

    for i in range(0, len(pcm), chunk):
        float_chunk = pcm[i:i + chunk]
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64_encode_audio(float_chunk),
        }
        await ws.send(json.dumps(payload))
        t_next += dur
        await asyncio.sleep(max(0, t_next - time.monotonic()))

    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

async def _recv_transcripts(ws, collected: List[str]) -> None:
    """Consumer: build current from streaming deltas."""
    current: List[str] = []

    try:
        async for msg in ws:
            ev = json.loads(msg)
            typ = ev.get("type")
            if typ == EV_DELTA:
                delta = ev.get("delta")
                if delta:
                    current.append(delta)
            elif typ == EV_DONE:
                collected.append("".join(current))
                current.clear()
                break
    except websockets.ConnectionClosedOK:
        pass

    if current:
        collected.append("".join(current))

def _session(model: str, vad: float = 0.5) -> dict:
    return {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "threshold": vad},
            "input_audio_transcription": {"model": model},
        },
    }

# ── New functions for real-time audio processing ────────────────────────────
def decode_audio_from_base64(audio_data: str) -> np.ndarray:
    """Decode base64 audio data to float32 numpy array."""
    try:
        # Decode base64 to bytes
        pcm_bytes = base64.b64decode(audio_data)

        # Convert PCM16 bytes to numpy array
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Convert to float32 range [-1.0, 1.0]
        pcm_float32 = pcm_int16.astype(np.float32) / 32767.0

        return pcm_float32
    except Exception as e:
        logger.error(f"Error decoding audio: {e}")
        return np.array([])

async def transcribe_audio_stream(audio_chunks: List[np.ndarray]) -> str:
    """Transcribe a stream of audio chunks using OpenAI's realtime API."""
    if not audio_chunks:
        return ""

    # Concatenate all audio chunks
    pcm = np.concatenate(audio_chunks)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    transcripts: List[str] = []

    try:
        async with websockets.connect(RT_URL, additional_headers=headers, max_size=None) as ws:
            # Send session configuration
            await ws.send(json.dumps(_session(MODEL_NAME)))

            # Send audio data
            await _send_audio(ws, pcm, CHUNK_SAMPLES, TARGET_SR)

            # Receive transcription results with timeout
            await asyncio.wait_for(_recv_transcripts(ws, transcripts), timeout=30.0)

    except asyncio.TimeoutError:
        logger.error("Timeout waiting for transcription results")
        raise HTTPException(status_code=500, detail="Transcription timeout")
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return " ".join(transcripts)

# ── WebSocket endpoint for real-time transcription ────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established for transcription")

    audio_buffer: List[np.ndarray] = []

    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)

            message_type = message.get("type")

            if message_type == "audio_chunk":
                # Decode and buffer audio data
                audio_data = message.get("audio")
                if audio_data:
                    decoded_audio = decode_audio_from_base64(audio_data)
                    if len(decoded_audio) > 0:
                        audio_buffer.append(decoded_audio)

                        # Send acknowledgment
                        await websocket.send_text(json.dumps({
                            "type": "chunk_received",
                            "buffer_length": len(audio_buffer)
                        }))

            elif message_type == "transcribe":
                # Process accumulated audio and return transcription
                if audio_buffer:
                    try:
                        transcription = await transcribe_audio_stream(audio_buffer)
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": transcription
                        }))
                        # Clear buffer after transcription
                        audio_buffer.clear()
                    except Exception as e:
                        logger.error(f"Transcription error: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Transcription failed: {str(e)}"
                        }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": ""
                    }))

            elif message_type == "clear_buffer":
                # Clear the audio buffer
                audio_buffer.clear()
                await websocket.send_text(json.dumps({
                    "type": "buffer_cleared"
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# ── HTTP endpoints ────────────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    """Serve the main HTML page."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Audio Transcription</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            justify-content: center;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .record-btn {
            background-color: #ff4757;
            color: white;
        }
        .record-btn:hover {
            background-color: #ff3338;
        }
        .record-btn.recording {
            background-color: #2ed573;
            animation: pulse 1.5s infinite;
        }
        .stop-btn, .transcribe-btn, .clear-btn {
            background-color: #5352ed;
            color: white;
        }
        .stop-btn:hover, .transcribe-btn:hover, .clear-btn:hover {
            background-color: #3742fa;
        }
        .stop-btn:disabled, .transcribe-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .status.connected {
            color: #2ed573;
        }
        .status.disconnected {
            color: #ff4757;
        }
        .transcription {
            background-color: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            min-height: 200px;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .transcription.empty {
            color: #6c757d;
            font-style: italic;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .info {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Real-time Audio Transcription</h1>

        <div class="info">
            <strong>Instructions:</strong> Click "Start Recording" to begin capturing audio,
            then click "Transcribe" to convert your speech to text using OpenAI's Whisper API.
        </div>

        <div class="status" id="status">Disconnected</div>

        <div class="controls">
            <button id="recordBtn" class="record-btn">Start Recording</button>
            <button id="stopBtn" class="stop-btn" disabled>Stop Recording</button>
            <button id="transcribeBtn" class="transcribe-btn" disabled>Transcribe</button>
            <button id="clearBtn" class="clear-btn">Clear</button>
        </div>

        <div class="transcription empty" id="transcription">
            Your transcription will appear here...
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let socket;
        let isRecording = false;

        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const clearBtn = document.getElementById('clearBtn');
        const status = document.getElementById('status');
        const transcription = document.getElementById('transcription');

        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            socket = new WebSocket(`${protocol}//${window.location.host}/ws/transcribe`);

            socket.onopen = function() {
                status.textContent = 'Connected';
                status.className = 'status connected';
            };

            socket.onclose = function() {
                status.textContent = 'Disconnected';
                status.className = 'status disconnected';
            };

            socket.onmessage = function(event) {
                const message = JSON.parse(event.data);

                if (message.type === 'transcription') {
                    if (message.text) {
                        transcription.textContent = message.text;
                        transcription.className = 'transcription';
                    }
                } else if (message.type === 'error') {
                    transcription.textContent = `Error: ${message.message}`;
                    transcription.className = 'transcription';
                }
            };
        }

        // Convert audio blob to base64 PCM16
        async function audioToBase64(blob) {
            return new Promise(async (resolve) => {
                try {
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                    // Get PCM data (mono channel)
                    const channelData = audioBuffer.getChannelData(0);

                    // Convert to 16-bit PCM
                    const pcm16 = new Int16Array(channelData.length);
                    for (let i = 0; i < channelData.length; i++) {
                        pcm16[i] = Math.max(-1, Math.min(1, channelData[i])) * 32767;
                    }

                    // Convert to base64
                    const pcmBytes = new Uint8Array(pcm16.buffer);
                    const base64 = btoa(String.fromCharCode(...pcmBytes));

                    audioContext.close();
                    resolve(base64);

                } catch (error) {
                    console.error('Error converting audio:', error);
                    resolve('');
                }
            });
        }

        // Start recording
        recordBtn.addEventListener('click', async function() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 24000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.start(100);
                isRecording = true;

                recordBtn.textContent = 'Recording...';
                recordBtn.className = 'record-btn recording';
                recordBtn.disabled = true;
                stopBtn.disabled = false;

            } catch (error) {
                console.error('Error accessing microphone:', error);
                alert('Error accessing microphone. Please make sure you have granted microphone permissions.');
            }
        });

        // Stop recording
        stopBtn.addEventListener('click', function() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;

                // Stop all tracks to release microphone
                mediaRecorder.stream.getTracks().forEach(track => track.stop());

                recordBtn.textContent = 'Start Recording';
                recordBtn.className = 'record-btn';
                recordBtn.disabled = false;
                stopBtn.disabled = true;
                transcribeBtn.disabled = false;
            }
        });

        // Transcribe audio
        transcribeBtn.addEventListener('click', async function() {
            if (audioChunks.length === 0) {
                alert('No audio recorded. Please record some audio first.');
                return;
            }

            transcribeBtn.disabled = true;
            transcribeBtn.textContent = 'Transcribing...';

            try {
                // Send all audio chunks to server
                for (let chunk of audioChunks) {
                    const base64Audio = await audioToBase64(chunk);
                    if (base64Audio) {
                        socket.send(JSON.stringify({
                            type: 'audio_chunk',
                            audio: base64Audio
                        }));
                    }
                }

                // Request transcription
                socket.send(JSON.stringify({
                    type: 'transcribe'
                }));

            } catch (error) {
                console.error('Error during transcription:', error);
                transcription.textContent = `Error: ${error.message}`;
            } finally {
                transcribeBtn.disabled = false;
                transcribeBtn.textContent = 'Transcribe';
            }
        });

        // Clear transcription
        clearBtn.addEventListener('click', function() {
            transcription.textContent = 'Your transcription will appear here...';
            transcription.className = 'transcription empty';
            audioChunks = [];
            transcribeBtn.disabled = true;

            if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({
                    type: 'clear_buffer'
                }));
            }
        });

        // Initialize WebSocket on page load
        initWebSocket();
    </script>
</body>
</html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "audio-transcription"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
