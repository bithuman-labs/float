ARG PYTHON_VERSION=3.13.1
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Install system dependencies
# libgl1 libglib2.0-0 
RUN apt-get update && \
    apt-get install -y --no-install-recommends libatomic1 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 && \
    chmod +x /usr/local/bin/dumb-init

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
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
CMD ["python", "agent.py", "start"]