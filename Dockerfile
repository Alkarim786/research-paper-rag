# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models PAPERRAG_DATA_DIR=/data PAPERRAG_INDEX_DIR=/data/index
WORKDIR /app

# CPU-only PyTorch keeps the image ~1 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY paperrag ./paperrag
COPY scripts ./scripts
RUN pip install ".[ml,llm]"

RUN useradd --create-home app && mkdir -p /data /models && chown -R app /data /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "paperrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
