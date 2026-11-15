FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY rtu-ai-assistant/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (UID 1000) required by Hugging Face Spaces
RUN useradd -m -u 1000 user

COPY --chown=user:user rtu-ai-assistant/backend/ ./rtu-ai-assistant/backend/
COPY --chown=user:user cleaned_dataset/ ./cleaned_dataset/

# Grant write permissions for SQLite database operations
RUN chmod -R 777 /app/rtu-ai-assistant/backend/data

USER user

WORKDIR /app/rtu-ai-assistant/backend
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
