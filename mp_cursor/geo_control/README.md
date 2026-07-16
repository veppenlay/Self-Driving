# 几何解耦控制探索（geo_control）

对应计划 `robust-lane-control-exploration`。把"控制"从"感知"里解耦：感知只估计可跨场景迁移的几何量
`e_y`（横向偏差）/`theta`（朝向），再用**源域冻结**的经典控制器算 `angular`。目标是提升跨场景（地下室）
鲁棒性，同时保持现有低成本自动标注，最终以**实车闭环**判成败。

## 铁律（第一性约束）
- **不针对目标场景调参**：阈值/控制增益只在源域 `dataset/26合并` 标定一次并冻结，`dataset/25地下室`
  只读复用。禁止在地下室 scan 参数"打赢"。
- **鲁棒性靠证据**：离线看**留出扰动**（`perturb.py`，纯合成、零地下室先验）下的衰减；上车看**冻结参数
  直接跑**的出界/干预。
- **纯经典 CV 不算鲁棒候选**（P2）：只做诊断（量化 CV 有多脆）+ 黄线丢失安全兜底。
- **数据边界**：train/val=`26合并`；test/闭环=`25地下室`；不覆盖锁存权重与 HMM。

## 文件角色
| 文件 | 作用 | 依赖 |
|---|---|---|
| `controller.py` | 冻结横向控制器 PID/Stanley/PurePursuit（`e_y,theta -> angular`） | numpy |
| `calibrate_controller.py` | 源域最小二乘拟合并冻结控制器增益 | numpy |
| `perturb.py` | 纯合成扰动（光照/对比度/模糊/噪声/丢帧），零目标域先验 | numpy |
| `augment.py` | 训练期几何保持的合成增强（鲁棒性核心杠杆） | numpy(+cv2 惰性) |
| `eval_harness.py` | 离线代理评测（感知保真/控制一致/jerk/coverage）+ 扰动衰减对比 | numpy |
| `geometry.py` / `seg_geometry.py` | `e_y/theta` 定义；掩码→车道中心→几何量 | numpy |
| `generate_geo_labels.py` | 自动派生 `theta` 标签（复用锁存 CV，双参考行） | cv2 |
| `models_affordance.py` / `datasets_geo.py` / `train_affordance.py` / `infer_affordance.py` | **P1** 中间表示：网络回归 `[steering,e_y,theta]` + 强增强，接冻结控制器 | torch+cv2 |
| `models_seg.py` / `datasets_seg.py` / `train_segmentation.py` / `infer_seg.py` | **P3** 分割：CV 黄掩码作自动伪标签 → 掩码 → 几何量 → 控制器 | torch+cv2 |
| `cv_perception.py` / `diagnose_cv.py` | **P2** 在线 CV 感知 + 脆弱度诊断 + 兜底源 | cv2 |
| `closed_loop_score.py` | 实车闭环打分与对比锁存基线 | numpy |
| `_smoke.py` | numpy-only 冒烟测试（本机可跑，无需 torch/cv2） | numpy |

车端：`epaicar_deploy/geometry_controller.py`（车上控制器），UDP server 与 ROS 节点新增
`--control-source geometry` / `~control_source:=geometry` 可切换来源，默认仍是锁存 `steering_hmm`。

## 流水线
本机（仅核心逻辑冒烟，不需 torch/cv2）：
```bash
python -m mp_cursor.geo_control._smoke
```

ve（GPU/dev）：
```bash
bash scripts/ve_geo_prepare.sh   # 1) 生成 theta 标签 + 冻结控制器
bash scripts/ve_geo_p1.sh        # 2) P1 训练 + 地下室 clean/扰动评测（首选）
bash scripts/ve_geo_p3.sh        # 3) P3 训练 + 评测（结构鲁棒升级，可选）
bash scripts/ve_geo_p2.sh        #    P2 CV 脆弱度诊断（仅诊断/兜底）
```
每个候选在 `mp_cursor/exp_<范式>/eval/` 产出 `metrics_clean.json` 与 `stress_degradation.json`。
据此 shortlist（看扰动衰减小、coverage 高、jerk 低）。

实车闭环（你执行；参数源域冻结、不重调）：
```bash
# 车端：几何控制驱动（默认 steering_hmm 不变）
roslaunch/rosrun ... epaicar_drive_ros.py _control_source:=geometry \
  _controller_json:=$(pwd)/epaicar_deploy/controller_pid.json
# 另开一个终端录遥测：
python3 epaicar_deploy/record_telemetry.py --port 15051 --out run_P1.csv
```
跑完按 `closed_loop_events_template.json` 手记事件，再打分对比锁存基线：
```bash
python -m mp_cursor.geo_control.closed_loop_score \
  --candidate-telemetry run_P1.csv --candidate-events events_P1.json \
  --baseline-events events_baseline.json \
  --output-json mp_cursor/exp_P1_affordance/closed_loop_verdict.json
```

## 判定
- 离线只用于 shortlist（代理与 CV 参考有循环性，非最终结论）。
- 最终成败：**冻结参数**下地下室闭环，`每圈出界/干预`不劣于锁存基线即视为更鲁棒的可用替代。
