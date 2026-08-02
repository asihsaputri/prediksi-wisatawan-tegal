FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads

ENV PORT=5000
EXPOSE 5000

CMD exec gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 300 \
    --preload
