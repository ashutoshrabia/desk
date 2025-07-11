FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy your prebuilt index + metadata + app
COPY news.index meta.json Articles.csv static/ ./static/ app.py ./

EXPOSE 7860

CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

