FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gosu \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/inspection

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/inspection-entrypoint

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev,realsense,runtime] && \
    chmod +x /usr/local/bin/inspection-entrypoint

ENTRYPOINT ["inspection-entrypoint"]
CMD ["bash"]
