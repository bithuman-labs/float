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
    uv pip install opencv-python-headless --no-deps && \
    uv pip uninstall livekit-plugins-bithuman && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/livekit/agents@longc/avatar-pause#subdirectory=livekit-plugins/livekit-plugins-bithuman && \
    uv pip uninstall livekit-agents && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/livekit/agents@longc/avatar-pause#subdirectory=livekit-agents

# Copy application code
COPY avatar_worker.py .
COPY dispatcher.py .



# Run the application
ENTRYPOINT ["python", "dispatcher.py", "--host", "0.0.0.0", "--port", "8089"]
