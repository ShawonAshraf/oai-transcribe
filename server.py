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

import numpy as np
import resampy
import soundfile as sf
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
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
            logger.info(f"Received OpenAI event: {typ}")

            if typ == EV_DELTA:
                delta = ev.get("delta")
                if delta:
                    current.append(delta)
                    logger.info(f"Transcription delta: '{delta}'")
            elif typ == EV_DONE:
                final_transcript = "".join(current)
                collected.append(final_transcript)
                current.clear()
                logger.info(f"Transcription completed: '{final_transcript}'")
                break
            elif typ == "error":
                error_info = ev.get("error", {})
                logger.error(f"OpenAI API error: {error_info}")
                raise Exception(f"OpenAI API error: {error_info.get('message', 'Unknown error')}")
            elif typ == "transcription_session.created":
                logger.info("Transcription session created successfully")
            elif typ == "transcription_session.updated":
                logger.info("Transcription session updated")
            elif typ == "input_audio_buffer.speech_started":
                logger.info("Speech started")
            elif typ == "input_audio_buffer.speech_stopped":
                logger.info("Speech stopped")
            elif typ == "input_audio_buffer.committed":
                logger.info("Audio buffer committed")
            elif typ == "conversation.item.created":
                logger.info("Conversation item created")
            elif typ == "response.created":
                logger.info("Response created")
            elif typ == "response.output_item.added":
                logger.info("Response output item added")
            elif typ == "response.output_item.done":
                logger.info("Response output item done")
            elif typ == "response.done":
                logger.info("Response done")
                # Also check if we have any transcription
                if current:
                    final_transcript = "".join(current)
                    collected.append(final_transcript)
                    logger.info(f"Final transcription from response.done: '{final_transcript}'")
                    break
            else:
                logger.info(f"Unhandled event type: {typ}, data: {ev}")

    except websockets.ConnectionClosedOK:
        logger.info("OpenAI WebSocket connection closed normally")
        # If we have any transcription data, save it
        if current:
            collected.append("".join(current))
            logger.info(f"Saved transcription on connection close: {''.join(current)}")
        pass
    except Exception as e:
        logger.error(f"Error in _recv_transcripts: {e}")
        raise

    # Final check for any remaining transcription
    if current:
        collected.append("".join(current))
        logger.info(f"Final transcription: {''.join(current)}")

def _session(model: str, vad: float = 0.5) -> dict:
    return {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            # Use a supported VAD type; to effectively disable VAD, you can omit this field
            "turn_detection": {"type": "server_vad", "threshold": vad},
            "input_audio_transcription": {"model": model},
        },
    }

# ── New functions for real-time audio processing ────────────────────────────
def decode_audio_from_base64(audio_data: str) -> np.ndarray:
    """Decode base64 PCM16 audio data to float32 numpy array."""
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_data)
        logger.info(f"Received audio data: {len(audio_bytes)} bytes")

        # Since we're now sending PCM16 directly from the client, decode as PCM16
        if len(audio_bytes) % 2 != 0:
            logger.error(f"Invalid PCM16 data: odd number of bytes ({len(audio_bytes)})")
            return np.array([])

        # Convert PCM16 bytes to int16 array
        pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 range [-1.0, 1.0]
        pcm_float32 = pcm_int16.astype(np.float32) / 32767.0
        
        logger.info(f"Successfully decoded PCM16: {len(pcm_float32)} samples")
        return pcm_float32

    except Exception as e:
        logger.error(f"Error decoding PCM16 audio: {e}")
        return np.array([])

async def transcribe_audio_stream(audio_chunks: List[np.ndarray]) -> str:
    """Transcribe a stream of audio chunks using OpenAI's realtime API."""
    if not audio_chunks:
        logger.warning("No audio chunks provided")
        return ""

    logger.info(f"Processing {len(audio_chunks)} audio chunks")
    for i, chunk in enumerate(audio_chunks):
        logger.info(f"Chunk {i}: {len(chunk)} samples")

    # Concatenate all audio chunks
    pcm = np.concatenate(audio_chunks)
    logger.info(f"Total audio length: {len(pcm)} samples ({len(pcm)/TARGET_SR:.2f} seconds)")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    transcripts: List[str] = []

    try:
        logger.info("Connecting to OpenAI realtime API...")
        async with websockets.connect(RT_URL, additional_headers=headers, max_size=None) as ws:
            logger.info("Connected to OpenAI realtime API")

            # Send session configuration
            session_config = _session(MODEL_NAME)
            logger.info(f"Sending session config: {session_config}")
            await ws.send(json.dumps(session_config))

            # Send audio data
            logger.info("Sending audio data to OpenAI...")
            await _send_audio(ws, pcm, CHUNK_SAMPLES, TARGET_SR)
            logger.info("Audio data sent, waiting for transcription...")

            # Receive transcription results with increased timeout
            logger.info("Waiting for transcription results...")
            await asyncio.wait_for(_recv_transcripts(ws, transcripts), timeout=120.0)
            logger.info(f"Transcription completed, result: '{''.join(transcripts)}'")

    except asyncio.TimeoutError:
        logger.error("Timeout waiting for transcription results (120 seconds)")
        # Don't fail completely, return empty result
        return "Transcription timeout - please try again"
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection closed: {e}")
        return "Connection to transcription service was lost"
    except Exception as e:
        logger.error(f"Error in transcription: {e}")
        return f"Transcription failed: {str(e)}"

    return " ".join(transcripts)

# ── WebSocket endpoint for real-time transcription ────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected for live transcription")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "OpenAI API key not configured"
        }))
        await websocket.close()
        return

    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    try:
        async with websockets.connect(RT_URL, additional_headers=headers, max_size=None) as openai_ws:
            logger.info("Connected to OpenAI realtime API")
            
            # Configure session
            session_config = _session(MODEL_NAME)
            await openai_ws.send(json.dumps(session_config))
            logger.info("Session configured")
            
            # Notify client that session is ready
            await websocket.send_text(json.dumps({
                "type": "session_started"
            }))

            # Task to forward OpenAI responses to client
            async def forward_openai_messages():
                try:
                    async for msg in openai_ws:
                        ev = json.loads(msg)
                        event_type = ev.get("type")
                        logger.info(f"OpenAI event: {event_type}")
                        
                        if event_type == EV_DELTA:
                            delta = ev.get("delta", "")
                            if delta:
                                await websocket.send_text(json.dumps({
                                    "type": "transcription_delta",
                                    "text": delta
                                }))
                                logger.info(f"Forwarded delta: '{delta}'")
                        elif event_type == EV_DONE:
                            await websocket.send_text(json.dumps({
                                "type": "transcription_final",
                                "text": ""
                            }))
                            logger.info("Transcription completed")
                        elif event_type == "error":
                            error_info = ev.get("error", {})
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": error_info.get("message", "OpenAI API error")
                            }))
                            logger.error(f"OpenAI error: {error_info}")
                        elif event_type == "transcription_session.created":
                            logger.info("Transcription session created")
                        elif event_type == "transcription_session.updated":
                            logger.info("Transcription session updated")
                        elif event_type == "input_audio_buffer.speech_started":
                            logger.info("Speech detected")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            logger.info("Speech ended")
                        elif event_type == "input_audio_buffer.committed":
                            logger.info("Audio buffer committed")
                        else:
                            logger.debug(f"Unhandled OpenAI event: {event_type}")
                            
                except websockets.ConnectionClosedOK:
                    logger.info("OpenAI WebSocket connection closed normally")
                except Exception as e:
                    logger.error(f"Error in OpenAI message forwarding: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Connection to OpenAI lost: {str(e)}"
                    }))

            # Start the OpenAI message forwarding task
            openai_task = asyncio.create_task(forward_openai_messages())

            try:
                # Handle client messages
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "audio_chunk":
                        # Forward audio chunk immediately to OpenAI
                        audio_data = message.get("audio")
                        if audio_data:
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_data
                            }))
                            logger.debug(f"Forwarded audio chunk: {len(audio_data)} base64 chars")
                        else:
                            logger.warning("Received empty audio chunk")

                    elif message_type == "commit":
                        # Commit the audio buffer to finalize transcription
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))
                        logger.info("Audio buffer committed")

                    elif message_type == "clear_buffer":
                        # Clear transcription state (client-side only for now)
                        await websocket.send_text(json.dumps({
                            "type": "buffer_cleared"
                        }))
                        logger.info("Buffer cleared")

            except WebSocketDisconnect:
                logger.info("Client disconnected")
                openai_task.cancel()
            finally:
                if not openai_task.done():
                    openai_task.cancel()

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Failed to connect to OpenAI: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Failed to connect to OpenAI Realtime API"
        }))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unexpected error: {str(e)}"
        }))

# ── HTTP endpoints ────────────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    """Serve the main HTML page."""
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        raise HTTPException(status_code=404, detail="index.html file not found")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "audio-transcription"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
