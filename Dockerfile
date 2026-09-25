# syntax=docker/dockerfile:1

# Ouroboros — application image for web UI runtime
# Usage:
#   docker build -f docker/Dockerfile.base -t ouroboros-base:local .
#   docker build -t ouroboros-web .
#   docker run --rm -p 8765:8765 ouroboros-web

ARG OUROBOROS_BASE_IMAGE=ouroboros-base:local
FROM ${OUROBOROS_BASE_IMAGE}

# Application environment
ENV APP_HOME=/app \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR ${APP_HOME}

# Install locked project dependencies separately so source edits reuse this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra browser --no-install-project

# Copy application
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra browser --no-editable

# Default environment
ENV OUROBOROS_SERVER_HOST=0.0.0.0 \
    OUROBOROS_SERVER_PORT=8765 \
    OUROBOROS_FILE_BROWSER_DEFAULT=${APP_HOME}

EXPOSE 8765

ENTRYPOINT ["python", "server.py"]
