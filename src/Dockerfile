# syntax=docker/dockerfile:1.7

# --- builder: resolve dependencies into a standalone venv ---------------------
FROM python:3.12-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# uv comes from PyPI rather than its own image: uv.lock stays the single source of
# truth, and this layer is thrown away with the builder stage anyway.
RUN pip install --no-cache-dir uv==0.9.15

WORKDIR /app

# Dependencies are installed from the lockfile alone, so this layer is reused
# until pyproject.toml or uv.lock actually change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

# --- runtime -----------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Non-root: the app only ever writes to /app/data.
RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app ./app
RUN mkdir -p /app/data && chown app:app /app/data

USER app
EXPOSE 8000
VOLUME ["/app/data"]

# No curl in the image, so the probe goes through the interpreter that is here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health').read()"

# One worker on purpose: sessions and their locks live in process memory,
# see the concurrency notes in README.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
