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
    uv pip uninstall livekit-plugins-bithuman && \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/longcw/agents@longc/cloud-bithuman-avatar#subdirectory=livekit-plugins/livekit-plugins-bithuman  && \
    uv pip uninstall opencv-python && \
    uv pip install opencv-python-headless --no-deps

# Copy application code
COPY agent.py .
COPY avatar.jpg .

# Ensure that any dependent models are downloaded at build-time
RUN python agent.py download-files

# Run the application
CMD ["python", "agent.py", "start"]