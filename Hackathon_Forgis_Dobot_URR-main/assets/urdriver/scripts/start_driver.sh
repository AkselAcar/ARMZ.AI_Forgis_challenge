#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:="${UR_TYPE:-ur3}" \
  robot_ip:="${ROBOT_IP:?Set ROBOT_IP in .env}" \
  launch_rviz:=false \
  headless_mode:=true
