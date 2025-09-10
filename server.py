import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import websockets
import logging
import json
from tts_service import TTSConfig
import formatting
from dictation_service import DictationService, DictationConfig

from settings import settings

import openai


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Audio Transcription Service")


# ── WebSocket endpoint for real-time transcription ────────────────────────────
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected for live transcription")

    api_key = settings.openai_api_key.get_secret_value()
    if not api_key:
        logger.error("OpenAI API key not set, closing websocket")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Error connecting to transcription service"
        },))

        await websocket.close()
        return

    headers = {"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"}

    try:
        config = DictationConfig()
        service = DictationService(config)

        async with websockets.connect(config.rt_url, additional_headers=headers, max_size=None) as openai_ws:
            logger.info("Connected to OpenAI realtime API")

            # Configure session
            dictation_session = service.create_session()
            await openai_ws.send(json.dumps(dictation_session))
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

                        if event_type == config.ev_delta:
                            delta = ev.get("delta", "")
                            if delta:
                                await websocket.send_text(json.dumps({
                                    "type": "transcription_delta",
                                    "text": delta
                                }))
                                logger.info(f"Forwarded delta: '{delta}'")
                        elif event_type == config.ev_done:
                            await websocket.send_text(json.dumps({
                                "type": "transcription_final",
                                "text": ""
                            }))
                            logger.info("Transcription completed")
                        elif event_type == "error":
                            error_info = ev.get("error", {})
                            error_code = error_info.get("code")
                            error_message = error_info.get("message", "OpenAI API error")
                            
                            # Handle empty buffer error gracefully - user stopped recording too quickly
                            if error_code == "input_audio_buffer_commit_empty":
                                logger.info("User stopped recording")
                                await websocket.send_text(json.dumps({
                                    "type": "transcription_final",
                                    "text": ""
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": error_message
                                }))
                                logger.error(f"OpenAI error: {error_info}")

                        else:
                            simple_event_logs = {
                                "transcription_session.created": "Transcription session created",
                                "transcription_session.updated": "Transcription session updated",
                                "input_audio_buffer.speech_started": "Speech detected",
                                "input_audio_buffer.speech_stopped": "Speech ended",
                            }

                            if event_type in simple_event_logs:
                                logger.info(simple_event_logs[event_type])
                            else:
                                logger.info(f"Unhandled OpenAI event: {event_type}")



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

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time text-to-speech.
    Receives text from the client and streams back Opus audio chunks from OpenAI.
    """
    await websocket.accept()
    print("WebSocket connection established.")

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    if not client:
        print("OpenAI client not initialized. Closing connection.")
        await websocket.close(code=1011, reason="OpenAI client not available")
        return

    config = TTSConfig()

    try:
        while True:
            # Receive text from the client
            text_to_speak = await websocket.receive_text()
            print(f"Received text: '{text_to_speak}'")

            text_to_speak = formatting.format_for_tts(text_to_speak)

            try:
                # Use OpenAI's streaming TTS API
                response = await client.audio.speech.create(
                    model=config.model_name,
                    voice=config.voice,
                    input=text_to_speak,
                    response_format=config.response_format, # Opus is great for web streaming
                )

                # Stream the audio chunks to the client
                print("Streaming audio to client...")
                # The response object has a streaming interface but iter_bytes() is not async
                # We need to use the synchronous iter_bytes method in an async context
                for chunk in response.iter_bytes(chunk_size=config.chunk_size):
                    await websocket.send_bytes(chunk)

                # Send an "end of stream" message to the client
                # This helps the client know when to finalize the audio playback.
                await websocket.send_text("EOS")
                print("End of stream sent.")

            except Exception as e:
                print(f"Error during OpenAI API call or streaming: {e}")
                # Inform the client of the error
                await websocket.send_text(f"ERROR: {e}")
                break # Break the loop on error

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("WebSocket connection closed.")



# ── HTTP endpoints ────────────────────────────────────────────────────────────
@app.get("/view/dictate")
async def view_dictate():
    """Serve the main HTML page."""
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        raise HTTPException(status_code=404, detail="index.html file not found")

@app.get("/view/tts")
async def view_tts():
    html_path = Path("tts.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        raise HTTPException(status_code=404, detail="tts.html file not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "audio-transcription"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
