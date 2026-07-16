#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record the inference UDP telemetry stream to CSV for closed-loop scoring.

The TRT UDP server sends JSON payloads (seq, stamp, raw, steering, angular, e_y, theta,
control_source) to a host:port. Bind this recorder to that port (e.g. run it, then point the
server / ROS node at the recorder port, or use a mirror port) to capture a run for offline
smoothness/saturation analysis by `mp_cursor/geo_control/closed_loop_score.py`.

Stdlib only (py3). Example:
  python3 record_telemetry.py --host 127.0.0.1 --port 15051 --out run_P1_basement.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time


FIELDS = ["recv_time", "seq", "stamp", "raw", "hmm", "steering", "angular", "e_y", "theta", "control_source"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Record inference UDP telemetry to CSV.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15051)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = until Ctrl-C).")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    start = time.time()
    count = 0
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        print("recording telemetry on {}:{} -> {}".format(args.host, args.port, args.out))
        try:
            while True:
                if args.duration > 0 and (time.time() - start) >= args.duration:
                    break
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                try:
                    if not isinstance(data, str):
                        data = data.decode("ascii", "ignore")
                    payload = json.loads(data)
                except Exception:
                    continue
                row = {k: payload.get(k, "") for k in FIELDS}
                row["recv_time"] = time.time()
                writer.writerow(row)
                count += 1
                if count % 100 == 0:
                    fh.flush()
        except KeyboardInterrupt:
            pass
    print("recorded {} rows".format(count), file=sys.stderr)


if __name__ == "__main__":
    main()
