"""
Part 6: Backend / Model Serving
-----------------------------------
FastAPI app exposing:
  POST /predict         - upload an audio clip, get emotion label + confidence
                           scores for every class
  WS   /predict-live     - scaffold for streaming/mic input: client streams
                           raw PCM16 chunks, server buffers ~3.5s and returns
                           a prediction, ready for more chunks after that
  GET  /health           - readiness probe (also reports which model backend
                           is active: cnn / rf / svm)
  GET  /emotions          - label list + display colors (mirrors ml/utils.py,
                           used by the mobile app / website for consistent theming)

The model(s) are loaded exactly once at process startup (see `lifespan`)
rather than per-request.
"""
import io
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))

from predict import get_predictor  # noqa: E402
from utils import EMOTION_COLORS, EMOTIONS  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ser-backend")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading emotion recognition model(s)...")
    t0 = time.time()
    try:
        predictor = get_predictor(prefer="cnn")
        logger.info(f"Model backend '{predictor.backend}' ready in {time.time()-t0:.2f}s")
        # Pay the one-time JIT/lazy-init cost now, not on a real request's dime -
        # see EmotionPredictor.warm_up()'s docstring for why this matters.
        warm_s = predictor.warm_up()
        logger.info(f"Warm-up inference done in {warm_s:.2f}s - ready for real traffic")
    except Exception as e:
        # Server still starts so /health can report the problem instead of crash-looping.
        logger.error(f"Model failed to load at startup: {e}")
    yield


app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Upload a short speech clip, get back the detected emotion and per-class confidence.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed mobile/web origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict
    model_used: str
    duration_ms: float


@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        return {"status": "ok", "model_backend": predictor.backend, "emotions": EMOTIONS}
    except Exception as e:
        return {"status": "model_unavailable", "detail": str(e)}


@app.get("/emotions")
def emotions():
    return {"emotions": EMOTIONS, "colors": EMOTION_COLORS}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model: str = None):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15 MB).")

    try:
        predictor = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    if model and model not in ("cnn", "rf", "svm"):
        raise HTTPException(status_code=400, detail="model must be one of: cnn, rf, svm")

    t0 = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        result = predictor.predict_from_path(tmp_path, model=model)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=422, detail=f"Could not process audio: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    result["duration_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.websocket("/predict-live")
async def predict_live(ws: WebSocket):
    """
    Scaffold for live/streaming inference.
    Client sends: JSON {"sr": 16000} once, then binary PCM16 mono chunks.
    Server buffers until it has ~3.5s of audio, runs a prediction, sends the
    result back as JSON, and keeps buffering for the next window. This is a
    simple fixed-window approach - swap in a sliding window / VAD trigger for
    a production streaming UX.
    """
    await ws.accept()
    predictor = get_predictor()
    sr = 16000
    window_samples = int(3.5 * sr)
    buffer = np.zeros((0,), dtype=np.float32)

    try:
        init = await ws.receive_json()
        sr = int(init.get("sr", 16000))
        window_samples = int(3.5 * sr)

        while True:
            chunk = await ws.receive_bytes()
            pcm16 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm16])

            if len(buffer) >= window_samples:
                window, buffer = buffer[:window_samples], buffer[window_samples:]
                result = predictor.predict_from_array(window, sr=sr)
                await ws.send_json(result)
    except WebSocketDisconnect:
        logger.info("predict-live client disconnected")
    except Exception as e:
        logger.exception("predict-live error")
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
