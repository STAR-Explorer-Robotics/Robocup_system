FROM ros:humble-ros-base-jammy

ARG ONNXRUNTIME_PACKAGE="onnxruntime>=1.18.0,<2.0"
ARG INSTALL_CUDA_PYTHON_LIBS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/inspection/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gosu \
    libglib2.0-0 \
    libgl1 \
    libgtk-3-0 \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-opencv \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-image-view \
    ros-humble-launch \
    ros-humble-launch-ros \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN mkdir -p ./inspection
COPY inspection/pyproject.toml ./inspection/pyproject.toml
COPY inspection/configs ./inspection/configs
COPY inspection/src ./inspection/src
COPY qrc_ros2_perception ./qrc_ros2_perception
COPY qrc_ros2_perception/docker-entrypoint.sh /usr/local/bin/qrc-ros2-perception-entrypoint

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    if [ "${INSTALL_CUDA_PYTHON_LIBS}" = "true" ]; then \
      python3 -m pip install --no-cache-dir \
        nvidia-cublas-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cudnn-cu12 \
        nvidia-cufft-cu12 \
        nvidia-curand-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-nvjitlink-cu12; \
    fi && \
    python3 -m pip install --no-cache-dir \
      "numpy>=1.26.0,<2.0" \
      "${ONNXRUNTIME_PACKAGE}" \
      "opencv-python>=4.9.0.80,<5.0.0" \
      "pydantic>=2.6.0" \
      "pupil-apriltags>=1.0.4" \
      "pyrealsense2>=2.55.1,<3.0" \
      "PyYAML>=6.0.1" \
      "pytest>=8.0.0" && \
    python3 -m pip install --no-cache-dir -e ./qrc_ros2_perception && \
    . /opt/ros/humble/setup.sh && \
    colcon build --base-paths qrc_ros2_perception --symlink-install && \
    mkdir -p /usr/local/share/ament_index/resource_index/packages && \
    mkdir -p /usr/local/share/qrc_ros2_perception/launch && \
    mkdir -p /usr/local/share/qrc_ros2_perception/configs && \
    mkdir -p /usr/local/lib/qrc_ros2_perception && \
    touch /usr/local/share/ament_index/resource_index/packages/qrc_ros2_perception && \
    cp qrc_ros2_perception/package.xml /usr/local/share/qrc_ros2_perception/package.xml && \
    cp qrc_ros2_perception/launch/*.launch.py /usr/local/share/qrc_ros2_perception/launch/ && \
    cp qrc_ros2_perception/configs/*.yaml /usr/local/share/qrc_ros2_perception/configs/ && \
    ln -sf /usr/local/bin/qrc-sensing-node /usr/local/lib/qrc_ros2_perception/qrc-sensing-node && \
    ln -sf /usr/local/bin/qrc-result-panel /usr/local/lib/qrc_ros2_perception/qrc-result-panel && \
    ln -sf /usr/local/bin/qrc-test-image-publisher /usr/local/lib/qrc_ros2_perception/qrc-test-image-publisher && \
    chmod +x /usr/local/bin/qrc-ros2-perception-entrypoint

WORKDIR /workspace/qrc_ros2_perception

ENTRYPOINT ["qrc-ros2-perception-entrypoint"]
CMD ["bash"]
