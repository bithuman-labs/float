FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies and configure opencv-python-headless
RUN uv sync --frozen --no-dev --no-install-project && \
    uv pip uninstall opencv-python opencv-python-headless && \
    uv pip install opencv-python-headless && \
    uv cache clean

# Set environment variables
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

COPY predownload.py ./
RUN python predownload.py

# temporally install livekit-agents from source
RUN apt-get update && apt-get install -y git && \
    uv pip uninstall livekit-agents && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/livekit/agents@main#subdirectory=livekit-agents && \
    rm -rf /var/lib/apt/lists/*

# copy models, checkpoints, and assets
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/
COPY assets/ ./assets/
COPY avatar_worker.py dispatcher.py generate.py ./

RUN python avatar_worker.py download-files

CMD ["python", "avatar_worker.py", "start"] 