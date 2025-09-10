import asyncio
import base64
import json
import logging
import os
import time

import resampy
import websockets
from fastapi import HTTPException
from pydantic import BaseModel
import struct
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
settings = Settings()


class DictationConfig(BaseModel):
    model_name: str = "gpt-4o-transcribe"
    target_sr : int = 24_000
    pcm_scale : int = 32_767
    chunk_samples : int = 3_072
    rt_url : str = "wss://api.openai.com/v1/realtime?intent=transcription"
    ev_delta : str = "conversation.item.input_audio_transcription.delta"
    ev_done : str = "conversation.item.input_audio_transcription.completed"


class DictationHelpers:
    @classmethod
    def float_to_16bit_pcm(cls, float32_array: np.ndarray):
        clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
        pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
        return pcm16

    @classmethod
    def base64_encode_audio(cls, float32_array: np.ndarray):
        pcm_bytes = cls.float_to_16bit_pcm(float32_array)
        encoded = base64.b64encode(pcm_bytes).decode('ascii')
        return encoded

    @classmethod
    async def send_audio(cls, ws, pcm: np.ndarray, chunk: int,) -> None:
        dur = 0.025
        t_next = time.monotonic()

        for i in range(0, len(pcm), chunk):
            float_chunk = pcm[i:i + chunk]
            payload = {
                "type": "input_audio_buffer.append",
                "audio": cls.base64_encode_audio(float_chunk),
            }
            await ws.send(json.dumps(payload))
            t_next += dur
            await asyncio.sleep(max(0, t_next - time.monotonic()))

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))



class DictationService:
    def __init__(self, config: DictationConfig):
        self.config = config

    def _session(self, vad: float = 0.5) -> dict:
        return {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                # Use a supported VAD type; to effectively disable VAD, you can omit this field
                "turn_detection": {"type": "server_vad", "threshold": vad},
                "input_audio_transcription": {"model": self.config.model_name},
            },
        }

    def load_and_resample(self, path: str) -> np.ndarray:
        sr = self.config.target_sr
        data, file_sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            data = resampy.resample(data, file_sr, sr)
        return data



    async def receive_transcripts(self, ws, collected: list[str]) -> None:
        """Consumer: build current from streaming deltas."""

        current: list[str] = []

        try:
            async for msg in ws:
                ev = json.loads(msg)
                typ = ev.get("type")
                logger.info(f"Received OpenAI event: {typ}")

                SIMPLE_LOG_EVENTS = {
                    "transcription_session.created",
                    "transcription_session.updated",
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_stopped",
                    "input_audio_buffer.committed",
                    "conversation.item.created",
                    "response.created",
                    "response.output_item.added",
                    "response.output_item.done",
                }

                if typ == self.config.ev_delta and (delta := ev.get("delta")):
                    current.append(delta)
                    logger.info(f"Transcription delta: '{delta}'")

                elif typ in [self.config.ev_done, "response.done"]:
                    # Handle both 'done' events in one block to avoid repeating logic.
                    logger.info("Response done" if typ == "response.done" else "Transcription completed")
                    if current:
                        final_transcript = "".join(current)
                        collected.append(final_transcript)
                        current.clear()
                        logger.info(f"Final transcription collected: '{final_transcript}'")
                        break

                elif typ == "error":
                    error_info = ev.get("error", {})
                    message = error_info.get('message', 'Unknown error')
                    logger.error(f"OpenAI API error: {error_info}")
                    raise Exception(f"OpenAI API error: {message}")

                elif typ in SIMPLE_LOG_EVENTS:
                    log_message = typ.replace("_", " ").replace(".", " ").capitalize()
                    logger.info(log_message)

                else:
                    # A catch-all for any other event types.
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


    async def transcribe_audio_stream(self, audio_chunks: list[np.ndarray]) -> str:
        """Transcribe a stream of audio chunks using OpenAI's realtime API."""
        if not audio_chunks:
            logger.warning("No audio chunks provided")
            return ""

        logger.info(f"Processing {len(audio_chunks)} audio chunks")
        for i, chunk in enumerate(audio_chunks):
            logger.info(f"Chunk {i}: {len(chunk)} samples")

        # Concatenate all audio chunks
        pcm = np.concatenate(audio_chunks)
        logger.info(f"Total audio length: {len(pcm)} samples ({len(pcm)/self.config.target_sr:.2f} seconds)")

        api_key = settings.openai_api_key
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

        transcripts: list[str] = []

        try:
            logger.info("Connecting to OpenAI realtime API...")
            async with websockets.connect(self.config.rt_url, additional_headers=headers, max_size=None) as ws:
                logger.info("Connected to OpenAI realtime API")

                # Send session configuration
                session_config = self._session()
                logger.info(f"Sending session config: {session_config}")
                await ws.send(json.dumps(session_config))

                # Send audio data
                logger.info("Sending audio data to OpenAI...")
                await DictationHelpers.send_audio(ws, pcm, self.config.chunk_samples)
                logger.info("Audio data sent, waiting for transcription...")

                # Receive transcription results with increased timeout
                logger.info("Waiting for transcription results...")
                await asyncio.wait_for(self.receive_transcripts(ws, transcripts), timeout=120.0)
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


