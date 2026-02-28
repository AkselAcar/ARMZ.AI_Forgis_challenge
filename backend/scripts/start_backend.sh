#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash

# Source DOBOT message bindings if present (built during image build)
if [ -f /dobot_msgs_ws/install/setup.bash ]; then
    source /dobot_msgs_ws/install/setup.bash
fi

uv run python src/main.py
