# Daily FIRMS ingest + 7-day forecast job (see docs/daily-ingest.md). Data stays in mounted volumes.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

USER app
# The DuckDB H3 and spatial extensions are installed at build time, so runs need no extension download.
RUN python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL h3 FROM community'); c.execute('INSTALL spatial')"

COPY src ./src

ENTRYPOINT ["python", "-m", "fire_atlas_data"]
CMD ["daily-ingest", "--help"]
