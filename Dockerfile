FROM python:3.11-slim AS base

WORKDIR /app
COPY pyproject.toml .
COPY lattice/ lattice/
RUN pip install --no-cache-dir ".[metrics]"

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz').raise_for_status()"

CMD ["uvicorn", "lattice.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
