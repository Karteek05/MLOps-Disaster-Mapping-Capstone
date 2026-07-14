# Build from the repo root: docker build -f backend/Dockerfile .
# (the app needs src/, params.yaml and models/ from outside backend/)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-serve.txt backend/requirements-serve.txt
RUN pip install --no-cache-dir -r backend/requirements-serve.txt

COPY src ./src
COPY backend ./backend
COPY params.yaml ./params.yaml
COPY models ./models

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
