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

COPY models/ ./models/
COPY checkpoints/ ./checkpoints/
COPY assets/ ./assets/
COPY predownload.py ./
COPY avatar_worker.py dispatcher.py generate.py ./

# Set environment variables
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

RUN python predownload.py
RUN python avatar_worker.py download-files

CMD ["python", "avatar_worker.py", "start"] 