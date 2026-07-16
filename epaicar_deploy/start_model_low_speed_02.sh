#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /opt/ros/melodic/setup.bash ]; then
  source /opt/ros/melodic/setup.bash
elif [ -f /opt/ros/kinetic/setup.bash ]; then
  source /opt/ros/kinetic/setup.bash
fi

if [ -f "$HOME/talos_ws/devel/setup.bash" ]; then
  source "$HOME/talos_ws/devel/setup.bash"
fi

if [ -f "$HOME/archiconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/archiconda3/etc/profile.d/conda.sh"
  conda activate pycuda
fi

export PYTHONUNBUFFERED=1

exec /usr/bin/python "$SCRIPT_DIR/epaicar_drive_ros_low_speed_02.py" \
  _engine_path:=/home/epaicar/results/0703_temporal3_2d_fp16.engine \
  _camera:=/dev/video1 \
  _start_infer_server:=true \
  _default_linear_speed:=0.2 \
  _slow_linear_speed:=0.16 \
  _angular_scale:=0.4 \
  _max_angular_z:=3.0 \
  _use_sign_file:=false \
  _use_yolo_topic:=false \
  _wait_for_start:=true \
  "$@"
