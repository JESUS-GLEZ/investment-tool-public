# =============================================================================
# Multi-stage Dockerfile for the Investment Tool
#
# Stages:
#   1. frontend-build — Builds the React SPA into static assets
#   2. python-dependencies — Locked dependencies and project in a virtualenv
#   3. production     — Runtime libraries, virtualenv, app and built frontend
#   4. nginx          — Nginx with baked-in frontend assets and reverse proxy config
#
# Usage:
#   docker build -t investment-tool .
#   docker compose -f docker-compose.prod.yml up -d
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build frontend static assets
# ---------------------------------------------------------------------------
FROM node:24.21.0-alpine@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS frontend-build

WORKDIR /build

COPY frontend/package.json frontend/package-lock.json frontend/.nvmrc ./
RUN test "$(node --version)" = "v$(cat .nvmrc)" \
    && npm ci --ignore-scripts --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2: Build Python dependencies separately from the runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS python-base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

FROM python-base AS python-dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir poetry==1.8.5 \
    && python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install Python dependencies (layer cached unless pyproject.toml changes)
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root \
    && rm -rf /root/.cache/pypoetry /root/.cache/pip

# Copy application source code
COPY src/ ./src/
COPY README.md ./

# Install the project itself (so `investment_tool` and `api` are importable)
RUN poetry install --only main --no-interaction --no-ansi \
    && rm -rf /root/.cache/pypoetry /root/.cache/pip


# ---------------------------------------------------------------------------
# Stage 3: Python runtime; no Poetry, compiler or package-download cache
# ---------------------------------------------------------------------------
FROM python-base AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libpq5 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
COPY --from=python-dependencies /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY alembic.ini README.md ./

# Copy built frontend assets from Stage 1
COPY --from=frontend-build /build/dist ./frontend/dist

# Copy entrypoints
COPY entrypoint/ ./entrypoint/

# Default data directory (mount or create as needed)
RUN mkdir -p /app/data/input /app/data/output

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI with uvicorn
# Override in docker-compose for Celery worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# ---------------------------------------------------------------------------
# Stage 4: Nginx — serves frontend static assets and proxies to the API
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS nginx

# Remove default Nginx site
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx config
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

# Copy built frontend assets from Stage 1
COPY --from=frontend-build /build/dist /usr/share/nginx/html
# Source exports can be owner-only; nginx's worker must read the built assets.
RUN chmod -R a+rX /usr/share/nginx/html

EXPOSE 80
