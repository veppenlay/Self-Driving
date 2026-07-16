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

exec /usr/bin/python "$SCRIPT_DIR/clockwise_turn_angular_test.py" "$@"
