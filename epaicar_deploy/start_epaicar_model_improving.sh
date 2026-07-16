#!/usr/bin/env bash
set -e

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

exec /usr/bin/python "$SCRIPT_DIR/epaicar_drive_ros.py" "$@"
