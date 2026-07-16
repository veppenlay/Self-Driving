# O2 跨场景鲁棒性候选交接

> 状态：准确性已确认，等待 EPAIcar 板端性能与完整链路验证  
> 数据协议：`dataset/26合并` 训练/验证，`dataset/25地下室` 固定外部测试  
> 日期：2026-07-15

## 1. 推荐方案

O2 使用两个结构完全相同的 Seq-CfC Temporal3 模型：

1. 模型 A：当前锁存 checkpoint。
2. 模型 B：从模型 A 以 `lr=1e-5` 续训，训练时冻结 MobileNetV3 backbone 的 BN running stats，其他参数正常更新。
3. 两模型对同一组三帧输入分别输出 steering；融合输出为：

   `steering = 0.75 * steering_A + 0.25 * steering_B`

4. 融合 steering 再进入现有、未改参数的 steering-only HMM。
5. `e_y` 不参与车端控制；如需保留日志，建议同样输出两模型原值，不要把它接入 HMM。

融合比例不是从地下室搜索得到。两个 seed 都按同一预设规则，仅在 26合并验证集上选择“达到至少 5% 验证改善的最小 alpha”，并独立得到 `alpha=0.25`。

## 2. 已确认结果

| 方案 | RAW MAE | HMM MAE | RAW >1 | HMM >1 | HMM >2 |
|---|---:|---:|---:|---:|---:|
| 锁存基线 | 0.204611 | 0.167248 | 26 | 28 | 6 |
| O2 seed 20260715 | 0.204509 | 0.161719 | 25 | 27 | 5 |
| O2 seed 20260716 | 0.203898 | 0.161678 | 26 | 27 | 5 |

相对锁存：

- 两个 seed 的 HMM MAE 分别改善 3.306% 和 3.330%。
- RAW MAE 分别改善 0.049% 和 0.348%，没有以 RAW 退化换取 HMM 收益。
- HMM `|error| > 1` 稳定从 28 降至 27；`|error| > 2` 稳定从 6 降至 5。
- 两次确认使用相同训练配置、相同 early stopping 和相同验证选参规则。

## 3. 远端复现产物

远端根目录：`/root/autodl-tmp/v-Net_cursor/mp_codex/runs/`

### seed 20260715

- E4-B checkpoint：`e4b_warm_frozen_backbone_bn_seed20260715/best_seq_cfc_temporal3_2d.pth`
- E4-B 训练摘要：`e4b_warm_frozen_backbone_bn_seed20260715/training_summary_2d.json`
- 融合选择：`o2_locked_e4b_output_ensemble/ensemble_selection.json`
- 最终比较：`o2_locked_e4b_output_ensemble/comparison.json`
- HMM 逐帧输出：`o2_locked_e4b_output_ensemble/hmm/current_postprocess_predictions.csv`

### seed 20260716

- E4-B checkpoint：`e4b_warm_frozen_backbone_bn_seed20260716/best_seq_cfc_temporal3_2d.pth`
- E4-B 训练摘要：`e4b_warm_frozen_backbone_bn_seed20260716/training_summary_2d.json`
- 融合选择：`o2_confirm_seed20260716/ensemble_selection.json`
- 最终比较：`o2_confirm_seed20260716/comparison.json`
- HMM 逐帧输出：`o2_confirm_seed20260716/hmm/current_postprocess_predictions.csv`

锁存模型与 HMM 文件保持未修改：

- `locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth`
- `epaicar_deploy/steering_only_hmm_online_model.json`

## 4. 板端资源判断

O2 的主要代价是同一输入执行两个约 4 MB 的 engine。现有单模型 H2D + execute + D2H 实测约 9.74 ms，因此串行双模型的纯推理粗估约 19.5 ms，即约 51 Hz，理论上仍高于 30 Hz。该数值不能替代完整相机—预处理—双模型—融合—HMM—ROS 控制链路实测。

板端必须验证：

1. 两个 engine 常驻时的峰值内存，目标 `< 3.2 GB`。
2. 连续 10 分钟完整链路的 P50/P95/P99 帧时和实际 FPS，P95 应满足 30 Hz。
3. 两模型必须复用同一份预处理后的三帧输入，禁止重复解码和重复 HSV/ROI/resize。
4. 融合必须发生在 RAW steering 后、HMM 前；alpha 固定为 0.25。
5. 离线 Python、ONNX 和 TensorRT 的融合前后输出误差分别保存，确认没有接口或量化漂移。
6. 保持当前 HMM JSON 哈希与参数不变。

## 5. LLM 下一步执行顺序

1. 为两个 checkpoint 分别导出 ONNX，逐模型做 PyTorch/ONNX parity。
2. 在 ve 上先模拟双模型串行推理，记录吞吐、显存和融合输出。
3. 获得用户对 EPAIcar 写入/运行授权后，再把候选写入独立目录；不得覆盖当前锁存 engine。
4. 在板端加载双 engine，复用输入 buffer，加入 `0.75/0.25` 融合并接现有 HMM。
5. 完成静态数据回放和完整链路性能验证后，才进入低速、可急停的实车闭环。
6. 若双模型完整链路达不到 30 Hz，不回头搜索 alpha；优先做共享 backbone 或把 O2 融合输出蒸馏回单模型。

## 6. 暂不执行

- 不补 R-DISC、T-DISC、E4-C 的消融。
- 不根据地下室继续搜索 alpha。
- 不修改 HMM 参数来放大收益。
- 不覆盖锁存 checkpoint、ONNX、engine 或 HMM。
- 在板端可行性确认前，不开展更多结构候选。
