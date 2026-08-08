"""Shared constants used across the ML pipeline and the FastAPI backend."""

EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

EMOTION_COLORS = {
    "angry": "#E5484D",
    "calm": "#4CC9B0",
    "disgust": "#8B7A3F",
    "fearful": "#7C6FE0",
    "happy": "#F5C542",
    "neutral": "#9AA5B1",
    "sad": "#4A7FC9",
    "surprised": "#E8823B",
}

MODEL_DIR_NAME = "models"
