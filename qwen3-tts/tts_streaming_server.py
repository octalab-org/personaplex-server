"""
Qwen3-TTS Streaming Server
===========================
HTTP server that provides OpenAI-compatible TTS endpoints with real streaming.
Uses the rekuenkdr/Qwen3-TTS-streaming fork for low-latency voice cloning.

Endpoints:
  POST /v1/audio/speech         - Standard TTS (streaming PCM/WAV chunks)
  POST /v1/audio/voice-clone    - Voice clone TTS (streaming with ref audio)
  GET  /health                  - Health check
  GET  /v1/audio/voices         - List available voices/languages

Runs on port 8880 by default (behind Nginx on 8998).
"""

import asyncio
import base64
import io
import json
import logging
import os
import struct
import sys
import tempfile
import time
import traceback
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from aiohttp import web

# Add parent to path for qwen_tts import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tts import Qwen3TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("tts-streaming")

# ─── Global State ─────────────────────────────────────────────────────────────
model: Optional[Qwen3TTSModel] = None
model_ready = False
model_lock = asyncio.Lock()

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8880"))
ENABLE_COMPILE = os.environ.get("TTS_ENABLE_COMPILE", "true").lower() == "true"
DECODE_WINDOW = int(os.environ.get("TTS_DECODE_WINDOW", "80"))

# Streaming defaults
DEFAULT_EMIT_EVERY = 12
DEFAULT_FIRST_CHUNK_EMIT = 5
DEFAULT_FIRST_CHUNK_WINDOW = 48
DEFAULT_FIRST_CHUNK_FRAMES = 48

# Supported languages
SUPPORTED_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "Spanish", "Portuguese", "German", "Italian", "Russian",
]


def load_model():
    """Load the Qwen3-TTS model with streaming optimizations."""
    global model, model_ready

    logger.info(f"Loading model: {MODEL_NAME}")
    start = time.time()

    torch.set_float32_matmul_precision("high")

    # Detect best available attention implementation
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Using FlashAttention2 for inference")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash-attn not available, falling back to SDPA (Scaled Dot Product Attention)")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")

    if ENABLE_COMPILE:
        try:
            logger.info("Enabling streaming optimizations (torch.compile + CUDA graphs)...")
            start = time.time()
            model.enable_streaming_optimizations(
                decode_window_frames=DECODE_WINDOW,
                use_compile=True,
                use_cuda_graphs=False,  # reduce-overhead includes CUDA graphs
                compile_mode="reduce-overhead",
            )
            elapsed = time.time() - start
            logger.info(f"Optimizations enabled in {elapsed:.1f}s")

            # Warmup run to trigger compilation
            logger.info("Running warmup generation...")
            start = time.time()
            warmup_chunks = list(model.stream_generate_voice_clone(
                text="Hello, this is a warmup test.",
                language="English",
                ref_audio=_generate_silence_wav(),
                ref_text="Warmup reference.",
                x_vector_only_mode=True,
                emit_every_frames=DEFAULT_EMIT_EVERY,
                decode_window_frames=DECODE_WINDOW,
                first_chunk_emit_every=DEFAULT_FIRST_CHUNK_EMIT,
                first_chunk_decode_window=DEFAULT_FIRST_CHUNK_WINDOW,
                first_chunk_frames=DEFAULT_FIRST_CHUNK_FRAMES,
            ))
            elapsed = time.time() - start
            logger.info(f"Warmup complete in {elapsed:.1f}s ({len(warmup_chunks)} chunks)")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without optimizations: {e}")
            logger.info("Model will run in eager mode (slower but stable)")

    model_ready = True
    logger.info("TTS model ready for streaming inference")


def _generate_silence_wav() -> str:
    """Generate a short silence WAV as base64 for warmup."""
    silence = np.zeros(24000, dtype=np.float32)  # 1 second of silence
    buf = io.BytesIO()
    sf.write(buf, silence, 24000, format="WAV")
    buf.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()


def _pcm_to_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Create a WAV header for streaming (unknown length → 0xFFFFFFFF)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,  # Unknown total size
        b"WAVE",
        b"fmt ",
        16,  # PCM format chunk size
        1,   # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,  # Unknown data size
    )
    return header


def _float32_to_int16(audio: np.ndarray) -> bytes:
    """Convert float32 PCM to int16 bytes."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    return audio_int16.tobytes()


def _detect_language(text: str) -> str:
    """Simple language detection based on character ranges."""
    for char in text:
        cp = ord(char)
        if 0x4E00 <= cp <= 0x9FFF:
            return "Chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "Japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "Korean"
        if 0x0400 <= cp <= 0x04FF:
            return "Russian"
        # Portuguese/Spanish/French accented chars
        if char in "àáâãçéêíóôõúüñ":
            return "Portuguese"
    return "English"


# ─── Handlers ─────────────────────────────────────────────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "ok" if model_ready else "loading",
        "model": MODEL_NAME,
        "streaming": True,
        "compile_enabled": ENABLE_COMPILE,
    })


async def handle_voices(request: web.Request) -> web.Response:
    """List supported voices/languages."""
    voices = [{"id": lang.lower(), "name": lang} for lang in SUPPORTED_LANGUAGES]
    return web.json_response({"voices": voices})


async def handle_speech(request: web.Request) -> web.StreamResponse:
    """
    POST /v1/audio/speech
    Standard TTS endpoint (OpenAI-compatible).
    Streams WAV audio chunks as they are generated.
    
    Body JSON:
      - input: text to synthesize (required)
      - voice: language hint (optional, default "Auto")
      - response_format: "wav" or "pcm" (default "wav")
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = body.get("input", "")
    if not text:
        return web.json_response({"error": "Missing 'input' field"}, status=400)

    voice = body.get("voice", "Auto")
    language = voice if voice in SUPPORTED_LANGUAGES else _detect_language(text)
    response_format = body.get("response_format", "wav")

    logger.info(f"[speech] text={text[:50]}... lang={language} format={response_format}")

    # Prepare streaming response
    resp = web.StreamResponse()
    if response_format == "pcm":
        resp.content_type = "audio/pcm"
    else:
        resp.content_type = "audio/wav"
    resp.headers["Transfer-Encoding"] = "chunked"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    try:
        async with model_lock:
            first_chunk = True
            for chunk, sr in model.stream_generate_voice_clone(
                text=text,
                language=language,
                ref_audio=None,
                ref_text=None,
                x_vector_only_mode=True,
                emit_every_frames=DEFAULT_EMIT_EVERY,
                decode_window_frames=DECODE_WINDOW,
                first_chunk_emit_every=DEFAULT_FIRST_CHUNK_EMIT,
                first_chunk_decode_window=DEFAULT_FIRST_CHUNK_WINDOW,
                first_chunk_frames=DEFAULT_FIRST_CHUNK_FRAMES,
            ):
                if first_chunk and response_format != "pcm":
                    await resp.write(_pcm_to_wav_header(sr))
                    first_chunk = False
                await resp.write(_float32_to_int16(chunk))

    except Exception as e:
        logger.error(f"[speech] Error: {e}\n{traceback.format_exc()}")

    await resp.write_eof()
    return resp


async def handle_voice_clone(request: web.Request) -> web.StreamResponse:
    """
    POST /v1/audio/voice-clone
    Voice clone TTS with streaming output.
    
    Body JSON:
      - text: text to synthesize (required)
      - language: language hint (optional, default "Auto")
      - ref_audio: base64-encoded reference audio OR URL (required)
      - ref_text: transcript of reference audio (optional, improves quality)
      - response_format: "wav" or "pcm" (default "wav")
      - stream: true/false (default true)
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    text = body.get("text", "")
    if not text:
        return web.json_response({"error": "Missing 'text' field"}, status=400)

    ref_audio = body.get("ref_audio", "")
    if not ref_audio:
        return web.json_response({"error": "Missing 'ref_audio' field"}, status=400)

    ref_text = body.get("ref_text", None)
    language = body.get("language", "Auto")
    response_format = body.get("response_format", "wav")
    stream = body.get("stream", True)

    # Determine x_vector_only_mode based on ref_text availability
    x_vector_only = ref_text is None or ref_text.strip() == ""

    logger.info(
        f"[voice-clone] text={text[:50]}... lang={language} "
        f"ref_text={'yes' if ref_text else 'no'} x_vector_only={x_vector_only} "
        f"stream={stream} format={response_format}"
    )

    if not stream:
        # Non-streaming: generate full audio and return
        try:
            async with model_lock:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                )
            buf = io.BytesIO()
            sf.write(buf, wavs[0], sr, format="WAV")
            buf.seek(0)
            return web.Response(
                body=buf.read(),
                content_type="audio/wav",
            )
        except Exception as e:
            logger.error(f"[voice-clone] Error: {e}\n{traceback.format_exc()}")
            return web.json_response({"error": str(e)}, status=500)

    # Streaming response
    resp = web.StreamResponse()
    if response_format == "pcm":
        resp.content_type = "audio/pcm"
    else:
        resp.content_type = "audio/wav"
    resp.headers["Transfer-Encoding"] = "chunked"
    resp.headers["Cache-Control"] = "no-cache"
    await resp.prepare(request)

    try:
        async with model_lock:
            first_chunk = True
            for chunk, sr in model.stream_generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                emit_every_frames=DEFAULT_EMIT_EVERY,
                decode_window_frames=DECODE_WINDOW,
                first_chunk_emit_every=DEFAULT_FIRST_CHUNK_EMIT,
                first_chunk_decode_window=DEFAULT_FIRST_CHUNK_WINDOW,
                first_chunk_frames=DEFAULT_FIRST_CHUNK_FRAMES,
            ):
                if first_chunk and response_format != "pcm":
                    await resp.write(_pcm_to_wav_header(sr))
                    first_chunk = False
                await resp.write(_float32_to_int16(chunk))

    except Exception as e:
        logger.error(f"[voice-clone] Error: {e}\n{traceback.format_exc()}")

    await resp.write_eof()
    return resp


async def handle_upload_audio(request: web.Request) -> web.Response:
    """
    POST /upload_audio/
    Upload reference audio for voice cloning (legacy compatibility).
    Stores the audio temporarily and returns a reference ID.
    """
    if not model_ready:
        return web.json_response({"error": "Model not ready"}, status=503)

    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "No file uploaded"}, status=400)

    # Save to temp file
    audio_bytes = await field.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    tmp.write(audio_bytes)
    tmp.close()

    # Create voice clone prompt
    try:
        async with model_lock:
            prompt = model.create_voice_clone_prompt(
                ref_audio=tmp.name,
                ref_text=None,
                x_vector_only_mode=True,
            )
        voice_id = f"clone_{int(time.time())}_{os.path.basename(tmp.name)}"
        # Store prompt in a simple dict (in production, use Redis/DB)
        if not hasattr(request.app, "voice_prompts"):
            request.app["voice_prompts"] = {}
        request.app["voice_prompts"][voice_id] = prompt

        return web.json_response({
            "voice_id": voice_id,
            "status": "success",
            "message": "Voice clone prompt created",
        })
    except Exception as e:
        logger.error(f"[upload] Error: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        os.unlink(tmp.name)


# ─── App Setup ────────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB max upload

    # Routes
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/audio/voices", handle_voices)
    app.router.add_post("/v1/audio/speech", handle_speech)
    app.router.add_post("/v1/audio/voice-clone", handle_voice_clone)
    app.router.add_post("/upload_audio/", handle_upload_audio)

    return app


if __name__ == "__main__":
    logger.info(f"Starting Qwen3-TTS Streaming Server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL_NAME}, Compile: {ENABLE_COMPILE}")

    # Load model synchronously at startup
    load_model()

    app = create_app()
    web.run_app(app, host=HOST, port=PORT, print=logger.info)
