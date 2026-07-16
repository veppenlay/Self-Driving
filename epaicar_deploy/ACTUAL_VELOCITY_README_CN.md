# EPAIcar 实际速度估计节点

车端底盘启动命令仍然是：

```bash
roslaunch eprobot_start AIcar_joy.launch
```

该 launch 会启动 `base_control`，底盘板通过 `/dev/ttyUSB0` 回传速度和航向信息，并发布：

- `/odom`：`nav_msgs/Odometry`，`twist.twist.linear.x` 为底盘板回传线速度 m/s，`twist.twist.angular.z` 为回传角速度 rad/s。
- `/imu_data`：`sensor_msgs/Imu`，`angular_velocity.z` 为同一底盘回传角速度 rad/s。
- `/feedback_vel`：`geometry_msgs/Twist`，只包含底盘回传线速度，可作为 `/odom` 失效时的备份。

新增节点 `epaicar_actual_velocity_estimator.py` 不控制小车，只订阅上述实测话题并发布：

- `/actual_velocity`：`geometry_msgs/TwistStamped`，推荐下游直接使用。
- `/actual_odom`：`nav_msgs/Odometry`，保留原 `/odom` 位姿，替换为滤波后的实际速度。
- `/actual_velocity_status`：`std_msgs/String`，JSON 状态，包含数据源、数据年龄和质量。

底盘驱动已做最小改动：复用 `/feedback_vel` 发布底盘回传的实际速度。

- `/feedback_vel.linear.x`：底盘回传线速度 m/s。
- `/feedback_vel.angular.z`：底盘 `0x0A` 帧回传角速度，按底层源码中的 `Vyaw * 0.01 * 0.823529` 转成 rad/s；比例来自 `L/B = 0.21/0.255`，可用 `~yaw_rate_feedback_scale` 覆盖。
- `/imu_raw_data`：由独立 `yesense_h30mini_imu_node.py` 发布 H30mini 实测 IMU 数据；`angular_velocity` 为 rad/s，`linear_acceleration` 为 m/s²。H30mini 官方示例中角速度单位为 dps，节点发布前已转换为 rad/s。

当前底层控制器源码是 `X4Diff` 四轮差速版本，`/cmd_vel.angular.z` 进入底板后被当作角速度命令用于左右轮差速。若 `/odom.twist.twist.angular.z` 和 `/imu_data.angular_velocity.z` 暂时为 0，节点会用最近的角速度命令兜底：

```text
angular_z = cmd_vel.angular.z
```

这个兜底不再乘线速度，因此键盘保持同一转向命令时，线速度变化不会改变估计角速度。键盘控制常见情况是命令不连续发布，因此默认保留最近 30 秒命令用于兜底；如果收到新的 0 转向命令，会立即按 0 处理。

运行：

```bash
cd /home/epaicar/talos_ws/src/e2e_v5/scripts/model_improving
./start_actual_velocity_estimator.sh
```

查看：

```bash
rostopic echo /actual_velocity
rostopic echo /actual_velocity_status
```

常用标定参数：

```bash
./start_actual_velocity_estimator.sh \
  _linear_scale:=1.0 \
  _linear_bias:=0.0 \
  _angular_scale:=1.0 \
  _angular_bias:=0.0 \
  _wheelbase:=0.335 \
  _use_feedback_steering:=false \
  _feedback_steering_deadband:=0.005 \
  _cmd_max_age:=30.0 \
  _cmd_angular_mode:=yaw_rate \
  _steering_scale:=1.0 \
  _max_steering_angle:=0.70
```

精度策略：

- 线速度优先使用 `/odom.twist.twist.linear.x`，来源是底盘板回传 `Vx/1000`。
- 角速度优先融合 `/imu_data.angular_velocity.z` 和 `/odom.twist.twist.angular.z`，必要时用 `/odom` 姿态 yaw 差分兜底。
- 当底盘回传角速度固定为 0，但键盘命令中有最近的非零角速度命令，则状态里的 `angular_source` 会显示 `cmd_yaw_rate`。
- 底盘驱动回退：用车端备份 `ai_racecar.py.codex_backup_<时间戳>` 覆盖回 `ai_racecar.py` 后重启 `AIcar_joy.launch`。
- 默认使用 3 点中值加 EMA 轻滤波，抑制串口抖动但保留 50 Hz 动态响应。
- 不使用 `/cmd_vel` 作为实际速度；`/cmd_vel` 只是控制命令。
