FROM ros:humble

RUN apt-get update && apt-get install -y \
    curl \
    ros-humble-ur \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY scripts/ ./scripts/
RUN sed -i 's/\r$//' scripts/*.sh && chmod +x scripts/*.sh


CMD ["bash"]