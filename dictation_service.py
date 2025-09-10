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

from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def session(self, vad: float = 0.5) -> dict:
        return {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                # Use a supported VAD type; to effectively disable VAD, you can omit this field
                "turn_detection": {"type": "server_vad", "threshold": vad},
                "input_audio_transcription": {"model": self.config.model_name},
            },
        }