#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/melodic/setup.bash
if [ -f /home/epaicar/talos_ws/devel/setup.bash ]; then
  source /home/epaicar/talos_ws/devel/setup.bash
fi

PORT_ARG=()
if [ -n "${H30MINI_PORT:-}" ]; then
  PORT_ARG=("_port:=${H30MINI_PORT}")
fi

exec python "${SCRIPT_DIR}/yesense_h30mini_imu_node.py" \
  "${PORT_ARG[@]}" \
  _baudrate:="${H30MINI_BAUDRATE:-460800}" \
  _frame_id:="${H30MINI_FRAME_ID:-IMU_link}" \
  _imu_topic:="${H30MINI_IMU_TOPIC:-/imu_raw_data}" \
  _status_topic:="${H30MINI_STATUS_TOPIC:-/h30mini_imu_status}" \
  _exclude_ports:="${H30MINI_EXCLUDE_PORTS:-/dev/ttyUSB0}" \
  _scan_seconds:="${H30MINI_SCAN_SECONDS:-1.5}" \
  "$@"
