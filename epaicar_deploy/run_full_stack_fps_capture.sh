#!/usr/bin/env bash
set -u

RUNTIME_SEC="${1:-75}"
SCRIPT_DIR="/home/epaicar/talos_ws/src/e2e_v5/scripts/model_improving"
LOG="/tmp/epaicar_full_stack_fps_$(date +%s).log"
UDP_PORT="15150"
NODE_NAME="epaicar_fps_test_${UDP_PORT}"

cd "$SCRIPT_DIR" || exit 1

setsid bash -lc "
source /opt/ros/melodic/setup.bash
if [ -f /home/epaicar/talos_ws/devel/setup.bash ]; then
  source /home/epaicar/talos_ws/devel/setup.bash
fi
source /home/epaicar/archiconda3/etc/profile.d/conda.sh
conda activate pycuda
export PYTHONUNBUFFERED=1
./start_epaicar_model_improving.sh \
  __name:=${NODE_NAME} \
  /cmd_vel:=/cmd_vel_fps_test \
  _udp_port:=${UDP_PORT} \
  _wait_for_start:=false \
  _auto_start_without_stdin:=false \
  _run_init_motion:=false \
  _use_sign_file:=false
" > "$LOG" 2>&1 &

PGID="$!"
echo "log=$LOG"
echo "pgid=$PGID"
echo "runtime_sec=$RUNTIME_SEC"

sleep "$RUNTIME_SEC"

kill -INT "-$PGID" 2>/dev/null || true
sleep 3
kill -TERM "-$PGID" 2>/dev/null || true
sleep 2
kill -KILL "-$PGID" 2>/dev/null || true

echo "---LOG---"
cat "$LOG"
echo "---PROC---"
ps -eo pid,ppid,stat,etime,cmd | grep -E "${NODE_NAME}|${UDP_PORT}|epaicar_temporal3_2d_trt_udp_server.py" | grep -v grep || true
