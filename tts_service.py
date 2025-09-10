from typing import Literal

from pydantic import BaseModel


class TTSConfig(BaseModel):
    model_name: str = "tts-1"
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "opus"
    chunk_size: int = 4096
    instructions: str = ("""
        You are a helpful assistant who reads the text supplied by user. 
        Reply in a slow and comfortable voice to give the user a sense of calm as if 
        someone is reading out a book or a story to them.
    """)

