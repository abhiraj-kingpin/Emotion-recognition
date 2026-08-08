FROM python:3.11-slim

# ffmpeg -> lets librosa decode mp3/m4a uploads, not just wav/flac
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY ml ./ml
COPY backend ./backend

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
