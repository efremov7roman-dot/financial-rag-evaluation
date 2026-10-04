FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FINANCIAL_RAG_DATA_DIR=/app/data \
    FINANCIAL_RAG_INDEX_DIR=/app/data/index/dense \
    FINANCIAL_RAG_FACT_INDEX_PATH=/app/data/index/xbrl/facts.jsonl \
    FINANCIAL_RAG_MODEL_CACHE_DIR=/app/data/models \
    FINANCIAL_RAG_API_HOST=0.0.0.0 \
    FINANCIAL_RAG_API_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["financial-rag", "serve"]
