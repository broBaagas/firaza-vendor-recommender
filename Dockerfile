# Image untuk inference API (FastAPI). Dipakai Hugging Face Spaces (sdk: docker, port 7860),
# tetapi bisa juga dijalankan di mana saja: docker build -t firaza-api . && docker run -p 7860:7860 firaza-api
FROM python:3.14-slim

# LightGBM butuh OpenMP (libgomp1) di Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces menjalankan container sebagai user UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# GEMINI_API_KEY dan DATABASE_URL diberikan lewat environment (Spaces: Settings > Secrets).
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
