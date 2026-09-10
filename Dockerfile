FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN addgroup --system aiworker && adduser --system --ingroup aiworker aiworker

COPY services/ai-service/pyproject.toml services/ai-service/README.md* ./
COPY --chown=aiworker:aiworker services/ai-service/app ./app
COPY --chown=aiworker:aiworker services/ai-service/migrations ./migrations
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

USER aiworker

EXPOSE 3009
CMD ["python", "-m", "uvicorn", "app.entrypoints.api:app", "--host", "0.0.0.0", "--port", "3009"]
