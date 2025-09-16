ARG PYTHON_VERSION=3.13.1
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install system dependencies
# libgl1 libglib2.0-0 
RUN apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create virtual environment
RUN uv venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt && \
    uv pip uninstall opencv-python && \
    uv pip install opencv-python-headless

# Use dev version of agents and plugins
ENV LIVEKIT_COMMIT=6994bf5d18de2c72a3e67fa2c70b61b80307d0d5
RUN uv pip uninstall livekit-agents livekit-plugins-bithuman && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps git+https://github.com/livekit/agents@${LIVEKIT_COMMIT}#subdirectory=livekit-agents && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps git+https://github.com/livekit/agents@${LIVEKIT_COMMIT}#subdirectory=livekit-plugins/livekit-plugins-bithuman


# Copy application code
COPY avatar_worker.py .
COPY dispatcher.py .
COPY agent_message_handler.py .
COPY avatar_actions.py .

# Ensure Python can find our modules
ENV PYTHONPATH=/app


# Run the application
ENTRYPOINT ["python", "dispatcher.py", "--host", "0.0.0.0", "--port", "8089"]
