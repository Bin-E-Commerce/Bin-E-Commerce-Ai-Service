FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY services/ai-service/pyproject.toml services/ai-service/README.md* ./
COPY services/ai-service/app ./app
COPY services/ai-service/migrations ./migrations
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 3009
CMD ["python", "-m", "uvicorn", "app.entrypoints.api:app", "--host", "0.0.0.0", "--port", "3009"]
