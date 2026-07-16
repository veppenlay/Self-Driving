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

export PYTHONUNBUFFERED=1

exec /usr/bin/python "$SCRIPT_DIR/epaicar_actual_velocity_estimator.py" \
  _odom_topic:=/odom \
  _imu_topic:=/imu_data \
  _feedback_vel_topic:=/feedback_vel \
  _cmd_vel_topic:=/cmd_vel \
  _actual_twist_topic:=/actual_velocity \
  _actual_odom_topic:=/actual_odom \
  _status_topic:=/actual_velocity_status \
  _rate_hz:=50.0 \
  _max_source_age:=0.30 \
  _cmd_max_age:=30.0 \
  _use_feedback_steering:=false \
  _feedback_steering_deadband:=0.005 \
  _kinematic_fallback_enabled:=true \
  _prefer_kinematic_when_commanded:=true \
  _cmd_angular_mode:=yaw_rate \
  _wheelbase:=0.335 \
  _steering_scale:=1.0 \
  _steering_bias:=0.0 \
  _max_steering_angle:=0.70 \
  _filter_window:=3 \
  _filter_alpha:=0.60 \
  "$@"
