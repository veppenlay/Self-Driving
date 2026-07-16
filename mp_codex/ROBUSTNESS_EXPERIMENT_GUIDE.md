# EPAIcar 单目端到端模型跨场景鲁棒性实验指导

> 版本：v1.2（快速发现优先，固定 26合并 -> 25地下室）  
> 日期：2026-07-15  
> 适用目录：`E:\桌面\v-Net\mp_codex`  
> 主要读者：负责实现、训练、评测和部署验证的 LLM / Codex 代理

## 0. 文档目的

本文档用于指导 LLM 在当前 v-Net 仓库中开展网络侧鲁棒性实验。执行目标是：

1. 保持当前锁存模型、数据边界和车端后处理不被覆盖。
2. 每个实验都以 `dataset/26合并` 为训练/内部验证数据，以 `dataset/25地下室` 为固定外部测试集。
3. 优先实验与当前网络互补、且适合 Jetson Nano 4GB 的轻量改动。
4. 每个实验都有明确的输入、输出、指标、准入门槛、回滚条件和证据文件。
5. 所有新增代码、配置和产物只写入 `mp_codex/`。

推荐执行主线：

```text
P0-Lite 本地工具链与筛选基线
  -> 快速候选 R：训练期车道结构辅助
  -> 快速候选 T：状态化、时间感知 CfC
  -> 固定预算初筛，选出前 1-2 个
  -> 完整预算确认；必要时组合 R + T
  -> 最终候选导出与板端验证
  -> 最终锁存
  -> 最后补消融、LOSO 诊断和 E2 安全增强
```

候选 R 和 T 可以在算力允许时独立并行；若只能串行，先执行候选 R，因为它不改变板端接口、验证成本更低。E4 只作为前两条路线均无稳定收益时的低成本救援方向。

### 0.1 快速发现策略

本项目当前目标是尽快找到有效改进，不是第一轮就完成论文级归因。采用三层证据，且三层都使用同一数据角色：`26合并/train.txt` 训练、`26合并/val.txt` early stopping、`26合并/test.txt` 所指向的 423 张 `25地下室` 图片做外部测试。

1. **初筛层**：单 seed、统一短训练预算，每个候选都输出 26合并验证集和 25地下室测试集指标，只判断是否存在强信号。
2. **确认层**：只对前 1-2 个候选使用完整训练预算复跑；候选接近或改善小于 3% 时才补多 seed。
3. **锁存层**：只对最终候选执行 ONNX / TensorRT、板端完整链路和实车验证。

第一轮允许使用一个经过明确设计的候选方案，不要求同时运行“去掉 delta_t”“不同 hidden size”“不同辅助权重”等消融。消融只在确认某条路线有效后补做。

---

## 1. LLM 执行总则

### 1.1 必须遵守

- 所有新增或修改内容必须位于 `mp_codex/`。
- 可以读取和导入 `locked_model/`、`epaicar_deploy/`、`dataset/` 中的内容，但不得覆盖锁存文件。
- 快速初筛可以使用一个围绕单一假设设计的完整候选，例如“状态化 + delta_t”或“空间辅助头 + 固定辅助权重”；无需先拆成全部消融。
- 不得在同一个快速候选中同时更换无关的 backbone、时序、损失、归一化和后处理。
- 所有结论必须由文件化指标支持，不能只根据训练日志或单张图片判断。
- RAW 和 HMM 指标必须同时报告。
- 必须报告尾部误差，不能只报告平均 MAE。
- 代码改动后必须运行最小可行测试，再开始完整训练。
- 每个实验都必须使用 `dataset/26合并` 训练/验证，并在 `dataset/25地下室` 上测试；不得悄悄改换数据版本或拆分。
- 初筛候选只需一个固定 seed 和统一短预算；前 1-2 名进入完整预算确认；只有最终候选进入导出和板端验证。
- 板端写入、文件清理、真实车辆控制属于后期步骤；执行前必须确认授权和安全条件。

### 1.2 禁止事项

- 不得修改或覆盖：
  - `locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth`
  - `epaicar_deploy/best_seq_cfc_temporal3_2d_correct.onnx`
  - `epaicar_deploy/steering_only_hmm_online_model.json`
  - `locked_model/evaluation/baseline/`
  - `locked_model/postprocess/`
- 不得让 `dataset/25地下室` 参与梯度更新、early stopping、学习率调度或同一实验内的超参数搜索；它只用于各实验结束后的固定外部测试和候选横向比较。
- 不得把源域随机验证 MAE 作为跨场景鲁棒性的唯一结论。
- 不得同时更换 backbone、时序结构、损失、归一化和后处理。
- 不得在相机链路不稳定时下结论说模型达不到 30 FPS。
- 不得仅因验证集更好就覆盖当前锁存方案。
- 不得在未 remap `/cmd_vel` 的情况下直接执行车辆闭环测试。

### 1.3 遇到阻塞时

LLM 应按以下顺序处理：

1. 保存当前命令、错误、环境版本和相关路径。
2. 判断是代码问题、数据边界问题、环境问题还是板端链路问题。
3. 先执行无副作用诊断。
4. 若需要删除板端文件、覆盖部署文件或启动车辆，停止并请求用户确认。
5. 不得用降低评测标准的方式绕过阻塞。

---

## 2. 当前锁存基线

### 2.1 网络与预处理

| 项目 | 锁存值 |
|---|---|
| 网络 | MobileNetV3-Small 多尺度融合 + 空间注意力池化 + Seq-CfC Temporal3 |
| 输入 | 连续 3 帧，HSV，底部 70% ROI，`144 x 192`，`frame_stride=1` |
| 输出 | `[steering, e_y]` |
| 在线控制 | 只使用 `steering`，不读取 `e_y` |
| 后处理 | `steering-only HMM online` |
| 训练 / 验证 | `dataset/26合并`，2654 / 569 |
| 固定外部测试 | `dataset/25地下室`，423；每个实验必须评测，不参与训练内调参 |
| RAW test MAE | `0.20461061395469163` |
| HMM test MAE | `0.1672482219835827` |
| RAW `|error| > 1.0` | 26 |
| HMM `|error| > 1.0` | 28 |
| HMM `|error| > 2.0` | 6 |

关键判断：

- 最佳源域验证 steering MAE 为 `0.1023256`。
- 固定 25地下室基准 RAW MAE 为 `0.2046106`，相对源域验证约扩大到 2 倍。
- HMM 将平均 MAE 降低约 18.3%，但 `|error| > 1.0` 的尾部大误差没有改善。
- 后续实验应优先改善跨场景衰减、尾部风险和丢帧恢复，而不是继续增加普通平滑。

### 2.2 当前模型已经覆盖的调研建议

当前模型已经具备：

- 轻量 MobileNetV3 backbone。
- 中层与末层特征融合。
- depthwise separable 融合块。
- 空间注意力池化。
- 真实三帧序列处理。
- CfC-lite 时序单元。
- `e_y` 辅助回归监督。
- FP16 TensorRT 部署。

因此以下方向不是当前优先项：

- 再添加普通 FPN / BiFPN，但不改变监督或时序。
- 从 CfC 简单切换为三帧 GRU。
- 只添加 SE / CBAM。
- 直接把序列从 3 帧增加到 5-8 帧。

### 2.3 事实来源

执行前必须读取以下文件：

```text
README.md
docs/locked_model_decision.md
docs/单目视觉端到端自动驾驶跨场景鲁棒性的网络侧优化报告.pdf
locked_model/README.md
locked_model/steering_models.py
locked_model/train.py
locked_model/evaluation/baseline/summary_test.json
locked_model/postprocess/summary_test_hmm.json
locked_model/labels/current/labels_2d_summary.json
epaicar_deploy/README.md
```

若这些文件与本文档中的数值发生冲突，以当前锁存决策和当前实际文件为准，并更新本文档版本。

### 2.4 每个实验的双基线要求

每个实验必须同时报告两类基线，缺一不可：

1. **当前最优锁存基线（绝对基线）**：checkpoint 为 `locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth`；25地下室 RAW MAE `0.20461061395469163`、固定 HMM MAE `0.1672482219835827`、RAW `|error| > 1.0` 为 26、HMM `|error| > 1.0` 为 28、HMM `|error| > 2.0` 为 6。
2. **同预算控制基线（公平性基线）**：使用当前最优模型结构、与候选完全相同的 train / val / test、seed、epoch、optimizer 和 checkpoint 选择规则重新训练。

当前基线文件 SHA256：checkpoint `9ADC49E5DB94E566C4B406F9BFEABF8FC40D11F4829F03A2DBB3D6C69AF295FD`；HMM JSON `319796A89B83765C744B95F0C96ABDE3B2D6279C29D0B831556632E3D4605F65`。每次实验开始时重新计算并与此处比对，漂移时停止并更新基线版本。

初筛先看候选相对同预算控制基线是否出现强信号；完整预算确认和最终锁存时，候选还必须优于当前最优锁存基线。不得用一个训练不足的控制组替代当前最优模型，宣称取得最终改进。

---

## 3. EPAIcar 板端约束

### 3.1 2026-07-14 实测资源

| 资源项 | 实测值 | 实验含义 |
|---|---:|---|
| 平台 | Jetson Nano Developer Kit | 不适合重型 Transformer |
| CPU / RAM | 4 核 / 4GB | 必须控制完整链路内存 |
| 模式 | MAXN | 板端基准必须记录功耗模式 |
| JetPack | R32.7.1 | 使用旧版 Jetson 软件栈兼容算子 |
| TensorRT | 8.2.x | 优先标准卷积、线性层和静态 shape |
| FP16 engine | 约 4MB | 参数规模不是当前瓶颈 |
| 纯执行 | 8.97ms，约 111.4 FPS | 仍有计算余量 |
| H2D + execute + D2H | 9.74ms，约 102.6 FPS | 原始 engine 满足 30Hz |
| 空闲可用内存 | 约 2.7GB | 推荐完整链路峰值 `< 3.2GB` |
| 根分区 | 37GB，100%，仅余约 249MB | 训练或部署前必须先释放空间 |

### 3.2 已知板端问题

- 板端远端推理服务脚本与仓库 `epaicar_deploy` 当前版本不一致。
- 板端服务不识别 `--hmm-model` 参数。
- 核验路径未发现当前锁存 HMM 文件。
- `/dev/video1` 显示为 `640 x 480 / YUYV / 30 FPS`，但本次 OpenCV 测试在首帧后阻塞。
- 当时 ROS master 未运行，完整 ROS 链路 FPS 未能测得。

因此 `102.6 FPS` 只能作为 TensorRT engine 基准，不能视为相机到控制的完整链路帧率。

---

## 4. 工作区与产物约定

### 4.1 目录结构

LLM 应维护以下结构：

```text
mp_codex/
  ROBUSTNESS_EXPERIMENT_GUIDE.md
  src/                          # 新模型、损失、数据协议、导出代码
  configs/                      # JSON / YAML 实验配置
  scripts/                      # 只服务本实验分支的入口脚本
  manifests/                    # 数据角色、文件 hash、板端环境清单
  runs/
    <experiment_id>/
      manifest.json
      config.json
      logs/
      checkpoints/
      evaluation/
        val_26merge/
        test_25basement/
        comparison_summary.json
      export/
        model.onnx
        parity_report.json
      deploy/
        engine_benchmark.json
        full_stack_benchmark.json
        board_manifest.json
  reports/
    comparison.csv
    comparison.md
    final_lock_report.md
```

### 4.2 实验编号

推荐格式：

```text
<family>_<variant>_<yyyymmdd>_<hhmmss>
```

示例：

```text
e1_stream_cfc_dt_h32_20260715_143000
e2_laplace_uncertainty_20260716_101500
```

### 4.3 每个 run 必须包含的 manifest

`manifest.json` 最少包含：

```json
{
  "experiment_id": "e1_stream_cfc_dt_h32_20260715_143000",
  "family": "E1",
  "hypothesis": "stateful dt-aware CfC improves cross-scene and frame-drop robustness",
  "baseline_checkpoint": "locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth",
  "baseline_checkpoint_sha256": "9ADC49E5DB94E566C4B406F9BFEABF8FC40D11F4829F03A2DBB3D6C69AF295FD",
  "hmm_sha256": "319796A89B83765C744B95F0C96ABDE3B2D6279C29D0B831556632E3D4605F65",
  "locked_baseline_metrics": {
    "test_set": "dataset/25地下室",
    "raw_mae": 0.20461061395469163,
    "hmm_mae": 0.1672482219835827,
    "raw_abs_error_gt_1": 26,
    "hmm_abs_error_gt_1": 28,
    "hmm_abs_error_gt_2": 6
  },
  "same_budget_baseline_run": "<experiment_id>",
  "git_commit": "<commit-or-null>",
  "dirty_worktree_summary": [],
  "data_protocol": "26merge_train_val__25basement_test",
  "train_list": "dataset/26合并/train.txt",
  "val_list": "dataset/26合并/val.txt",
  "test_list": "dataset/26合并/test.txt",
  "test_root": "dataset/25地下室",
  "split_hashes": {
    "train": "<sha256>",
    "val": "<sha256>",
    "test": "<sha256>"
  },
  "seeds": [20260715],
  "changed_factors": ["stateful_hidden", "delta_t"],
  "fixed_factors": ["HSV", "144x192", "bottom_70pct_ROI", "locked_HMM"],
  "status": "planned"
}
```

`changed_factors` 中出现两个以上主要因素时，LLM 必须解释为什么不能拆分。

---

## 5. 每个实验的固定数据协议

### 5.1 数据来源

用户所称“26-合并”在仓库中的实际路径是 `dataset/26合并`。它是所有实验唯一允许的训练/内部验证集合，包含四个来源：

| 来源 | 总图片数 |
|---|---:|
| 25杭州 | 1743 |
| 25茂华 | 1151 |
| 26北理工l | 399 |
| 26北理工r | 499 |

`dataset/25地下室` 是所有实验固定的外部测试集，共 423 张。仓库中的 `dataset/26合并/test.txt` 已指向这 423 张图片；实际运行前必须验证该映射，不能误用 `dataset/25地下室_2` 或 `dataset/前面段/25地下室`。

### 5.2 强制数据角色

| 角色 | 固定输入 | 当前数量 | 允许用途 |
|---|---|---:|---|
| 训练 | `dataset/26合并/train.txt` | 2654 | 梯度更新 |
| 内部验证 | `dataset/26合并/val.txt` | 569 | early stopping、checkpoint 选择、学习率调度 |
| 外部测试 | `dataset/26合并/test.txt` + `dataset/25地下室` | 423 | 每个实验结束后的固定测试、候选横向比较 |

每个基线、R-DISC、T-DISC、组合实验、救援实验和最终候选都必须遵守这张表。若任何列表、根目录或样本数量改变，必须新建实验族，重跑同预算基线，不得与旧结果直接比较。

2026-07-15 已核对的列表 SHA256：

- train：`2AB659B3BC75004FD8CC56945463182950DFB77140432CC5EC6560933ECB4196`
- val：`4E7D5DC7CD9FAE1DC0BAD98ED45DAAA4C91190CEDC068573D028F5C8618CFC6B`
- test：`6371392AD8B6DDEE92CD5CC0301BCFB0484613F745BA603D868345C66059915B`

运行时仍须重新计算完整 hash，防止列表在后续实验中发生漂移。

当前 `train_clean.txt`、`val_clean.txt`、`test_clean.txt` 分别与对应非 `_clean` 文件逐字节同 hash。现有脚本若使用 `_clean` 名称可以继续，但 manifest 必须记录实际文件名并验证 hash 相同；一旦不同，禁止混用结果。

### 5.3 每次运行前的自动检查

1. 计算 `train.txt`、`val.txt`、`test.txt` 的 SHA256 并写入 manifest。
2. 从每行**最右侧数值标签**向左切分路径，再规范化为绝对路径；路径中可能有空格，禁止直接取空格分割后的第一列。
3. 断言 train / val / test 三者没有重复图片。
4. 断言训练和验证图片都位于 `dataset/26合并/<四个来源>/`。
5. 断言测试图片全部位于 `dataset/25地下室/`，数量为 423。
6. 断言测试路径不含 `25地下室_2` 和 `前面段`。
7. 保存检查结果为 `evaluation/data_leakage_check.json`；任一断言失败则停止训练。

### 5.4 25地下室使用边界

25地下室会被每个实验重复评测，因此它是本实验分支的**固定外部基准测试集**，不再声称是只看一次的“完全未触碰最终测试集”。必须遵守：

- 不参与梯度、early stopping、scheduler 或 checkpoint 选择。
- 不根据单次地下室曲线继续搜索 hidden size、reset threshold、`lambda_aux`、`log_sigma clamp` 或 HMM 参数。
- 每个实验只评测预先由 26合并验证集选出的 checkpoint，不得在多个 checkpoint 中挑地下室最优者。
- 地下室 RAW、固定 HMM、尾部误差和逐帧预测必须完整保留，不能只报告有利指标。
- 若未来需要发表级无偏泛化结论，应另采一个从未参与实验选择的新场景；本阶段不以此阻塞快速探索。

### 5.5 LOSO 的位置

四来源 LOSO 不再是首轮候选或每个实验的必做项。某条路线在 25地下室上确认有效后，可在最后补做 LOSO，作为收益来源和场景依赖性的诊断；它不阻塞候选发现，也不替代上述固定训练/测试协议。

---

## 6. 固定评价指标

### 6.1 点误差

- RAW steering MAE。
- HMM steering MAE。
- RMSE。
- P50 / P90 / P95 / P99 absolute error。

### 6.2 尾部风险

- `abs_error > 0.5` 的数量和占比。
- `abs_error > 1.0` 的数量和占比。
- `abs_error > 2.0` 的数量和占比。
- 最大绝对误差及对应帧。

### 6.3 时序质量

- `abs(pred_t - pred_t-1)` 的均值和 P95。
- 丢帧后恢复到正常误差范围所需帧数。
- 状态 reset 前后的控制突变量。
- 有实车速度时报告 jerk 或 angular acceleration。

### 6.4 跨场景质量

- 25地下室 RAW / 固定 HMM MAE 相对同预算基线的改善百分比。
- 25地下室 P95 / P99 和 `|error| > 0.5 / 1.0 / 2.0` 的变化。
- 26合并验证集与 25地下室测试集的泛化差距。
- 最后补做 LOSO 时，再报告 macro、worst-scene 和标准差。

### 6.5 部署质量

- ONNX size。
- TensorRT engine size。
- execute-only mean / P90 / P99 latency。
- H2D + execute + D2H mean / P90 / P99 latency。
- 相机完整链路实际 FPS。
- 峰值 RAM、CPU、GR3D、EMC 和温度。

### 6.6 统一准入门槛

| 指标 | 最低门槛 | 推荐目标 |
|---|---:|---:|
| 初筛 25地下室 HMM MAE | 相对同预算基线下降 >= 3% | 下降 >= 5% |
| 完整预算 25地下室 HMM MAE | `< 0.167248` | `<= 0.159` |
| 地下室 `|error| > 1.0` | `< 28` | `<= 22` |
| 26合并验证 MAE | 不明显退化（<= 3%） | 同时改善 |
| 完整链路 | 稳定 `>= 30 FPS` | P99 周期 `< 25ms` |
| 板端内存 | 峰值 `< 3.4GB` | 峰值 `< 3.2GB` |

初筛时：地下室 HMM MAE 改善 `>= 5%` 且 `|error| > 1.0` 数量不增加，直接晋级；改善 2%-5% 记为弱信号，只有资源允许或 26合并验证集同步改善时晋级；退化 `>= 3%` 直接淘汰。完整预算确认仍必须相对一个使用完全相同预算和数据协议的基线比较。

完整链路和板端内存门槛只在 S3 唯一候选阶段生效，不能以“尚未上板”为由阻塞 S0-S2。

---

## 7. P0：实验前置条件

P0 分为两部分：`P0-Lite` 是开始候选训练前的硬门；`P0-Deploy` 可与离线探索并行，不得阻塞候选训练，但必须在模型导出、板端写入或实车测试前完成。

### 7.1 P0-Lite：本地训练与评测基线

在开始 R-DISC / T-DISC 前必须完成：

1. 运行第 5.3 节的数据角色和泄漏检查。
2. 加载锁存 checkpoint，完成单 batch forward，确认输出 `[steering, e_y]`、预处理和标签尺度一致。
3. 用当前结构、`dataset/26合并/train.txt`、`val.txt`、固定 seed 和初筛短预算训练一个 `S0-short-baseline`。
4. 在 `dataset/25地下室` 上输出 RAW、固定 HMM、分位数和尾部计数，作为所有快速候选的同预算比较基线。
5. 保存训练配置、最佳 checkpoint 的选择依据、三份列表 hash 和逐帧预测。

锁存模型已有地下室参考值为 RAW MAE `0.20461061395469163`、HMM MAE `0.1672482219835827`。这些值用于发现工具链或结果量级异常；快速候选的晋级必须优先相对 `S0-short-baseline` 判断，不能把不同训练预算直接比较。

### 7.2 P0-Deploy-1：板端存储

- 只读列出大文件和候选清理路径。
- 不自动删除未知文件。
- 请求用户确认后再清理。
- 目标根分区可用空间 `>= 3GB`，推荐 `>= 5GB`。

### 7.3 P0-Deploy-2：部署文件一致性

以下三个产物视为同一版本单元：

```text
best_seq_cfc_temporal3_2d_correct.onnx
epaicar_temporal3_2d_trt_udp_server.py
steering_only_hmm_online_model.json
```

LLM 必须：

1. 计算开发机 SHA256。
2. 只读计算板端 SHA256。
3. 输出差异清单。
4. 未经授权不覆盖板端文件。
5. 同步后再次确认板端服务支持 `--hmm-model`。

### 7.4 P0-Deploy-3：相机与完整链路

- 独立读取 `/dev/video1` 至少 10 分钟。
- 记录实际 FPS、最长帧间隔、丢帧数和重连数。
- 不能只验证 `v4l2-ctl` 声明的 30 FPS。
- 模型进程必须对 `cap.read()` 超时可退出或重连。
- 相机重连后必须重置未来的 streaming hidden state。
- 首先使用独立 UDP 端口，不连接真实 `/cmd_vel`。

### 7.5 P0 完成证据

写入：

```text
mp_codex/runs/p0_baseline/
  baseline_summary.json
  data_leakage_check.json
  split_hashes.json
  file_hashes.json
  board_environment.json
  camera_stability.json
  engine_benchmark.json
  full_stack_benchmark.json
```

---

## 8. E1：状态化、时间感知 CfC

### 8.1 假设

当前 `RegressionSequenceCfCSteeringNet` 存在两个限制：

1. 每次调用重新处理三帧，并从零 hidden state 开始。
2. `_CfCLiteCell` 没有显式 `delta_t`，无法根据帧间隔调整状态更新。

E1 假设：跨推理周期保留 hidden state，并输入 `delta_t`，可以提高跨场景连续性、丢帧恢复和控制平滑性，同时减少每周期 backbone 计算。

### 8.2 代码边界

- 新增代码放在 `mp_codex/src/`。
- 不修改 `locked_model/steering_models.py`。
- 可通过 import、继承或显式复制最小必要模块复用锁存结构。
- 若复制锁存结构，必须记录来源文件和 commit / hash。

建议文件：

```text
mp_codex/src/streaming_cfc.py
mp_codex/src/sequence_dataset.py
mp_codex/scripts/train_e1.py
mp_codex/scripts/export_e1_onnx.py
mp_codex/configs/e1_*.json
```

### 8.3 模型接口契约

```text
inputs:
  image_t      float32 [B, 3, 144, 192]
  hidden_prev  float32 [B, H]       # H = 32 or 64
  delta_t      float32 [B, 1]       # seconds, clamp to [0.0, 0.5]

outputs:
  steering_ey  float32 [B, 2]
  hidden_next  float32 [B, H]
```

部署状态必须显式作为 TensorRT binding 输入输出，不能只存在于 Python 全局变量中。

### 8.4 推荐实现

1. 当前帧只执行一次 MobileNetV3 backbone。
2. 复用当前多尺度融合和空间注意力池化。
3. 对视觉特征执行 `temporal_projection`。
4. 将 `delta_t` 编码为 4-8 维向量，再与视觉投影融合。
5. 使用显式 `hidden_prev` 更新 CfC state。
6. 返回 `hidden_next`。
7. 相机重连、人工停车、模型异常或超时后清零 hidden。

### 8.5 训练协议

- 快速候选固定为 `T-DISC`：Streaming CfC，`H=32`，显式 `delta_t`，片段长度 8。
- 每个片段起点 hidden 置零。
- 使用 truncated BPTT。
- 保持 HSV、ROI、输入尺寸、steering / e_y 目标与基线一致。
- 不先引入新的视觉增强或新损失。
- 若图片没有真实 timestamp，正常帧间隔先设为 `1/30s`；延迟扰动只用于指定消融，并记录生成规则。
- 初筛阶段不搜索长度、hidden size 或 reset threshold；只按统一短预算与 `S0-short-baseline` 比较。

### 8.6 后置消融矩阵

| 编号 | 结构 | 长度 | 主要变量 |
|---|---|---:|---|
| E1-A | 当前三帧 stateless CfC | 3 | 基线 |
| E1-B | Streaming CfC，H=32，无 `delta_t` | 8 | 仅状态化 |
| E1-C | Streaming CfC，H=32，含 `delta_t` | 8 / 12 | 时间感知 |
| E1-D | Streaming CfC，H=64，含 `delta_t` | 12 / 16 | 容量上限 |

首轮直接运行 E1-C 的固定配置 `T-DISC`，不先运行 E1-B / E1-D。只有 T-DISC 在完整预算确认有效后，才补 E1-A/B/C 用于归因；只有明确存在容量不足证据时才执行 E1-D。

### 8.7 丢帧与 reset 测试

初筛只做正常 30Hz、一次单帧跳过和一次 reset smoke test。T-DISC 晋级完整预算后，再完成以下全部测试：

- 正常 30Hz：`delta_t=33ms`。
- 每 30 帧随机跳过 1 帧。
- 随机连续跳过 2-3 帧。
- 延迟阶跃：100ms、200ms、500ms。
- reset threshold：150ms、350ms、500ms。

报告：

- 丢帧前后 MAE。
- 恢复到正常误差范围所需帧数。
- reset 前后 steering 跳变量。
- hidden norm 的时间序列。

### 8.8 E1 准入

推荐候选 T-DISC 必须满足当前阶段门槛：

- 初筛时 25地下室 HMM MAE 相对 `S0-short-baseline` 下降 `>= 5%` 可直接晋级，2%-5% 仅视为弱信号。
- 25地下室 `|error| > 1.0` 数量不增加，26合并验证 MAE 不退化超过 3%。
- 连续丢 2 帧后的恢复时间优于 E1-A。
- 状态 reset 后无明显控制突跳。
- ONNX / TensorRT parity 和完整链路 30 FPS 仅在 T-DISC 成为最终候选后检查，不阻塞首轮发现。

### 8.9 E1 回滚

出现以下任一条件则停止 E1 扩展：

- 25地下室 HMM MAE 相对同预算基线恶化 `>= 3%`，或尾部错误明显增加。
- hidden 在直道持续漂移。
- 相机重连或 reset 后产生大角度突跳。
- TensorRT 多 binding 支持不稳定。
- P99 完整周期超过 25ms。

---

## 9. E2：转向不确定性头（有效候选之后）

### 9.1 前置条件

- R-DISC、T-DISC 或其组合已经选出一个准确性候选。
- E2 的主要目标是风险识别和安全降级，不是最快降低 MAE，因此不进入首轮候选竞赛。
- 不得一边更换主结构一边调 E2 损失。

### 9.2 假设

HMM 改善平均误差但没有改善尾部风险。增加一个 aleatoric uncertainty 标量，可以识别当前视觉输入或状态不可信的帧，并支持限速、保持或降低 HMM 跟随。

### 9.3 输出契约

```text
[steering_mean, e_y, log_sigma]
```

推荐从 Laplace NLL 开始：

```text
L_steer = abs(y - mean) * exp(-log_sigma) + log_sigma
L_total = L_steer + lambda_ey * L_ey
log_sigma = clamp(log_sigma, -5, 2)
```

### 9.4 后置安全消融矩阵

| 编号 | 方法 | 用途 |
|---|---|---|
| E2-A | 确定性 steering + 当前 HMM | 基线 |
| E2-B | Laplace NLL + `log_sigma` | 风险排序和校准 |
| E2-C | E2-B + uncertainty-aware HMM | 高不确定性时降低滤波跟随 |
| E2-D | E2-B + 限速 / 保持策略 | 实车安全降级 |

有效准确性候选确定后，只先运行 E2-B；E2-B 通过前，不执行 E2-C / E2-D。

### 9.5 指标

- RAW / HMM MAE。
- NLL。
- 分箱 predicted sigma 与实际 absolute error。
- 以 `|error| > 0.5` 和 `> 1.0` 为正样本的 AUROC / AUPRC。
- risk-coverage curve。
- 高风险帧连续触发长度。
- 误限速比例。
- 危险大误差召回率。

### 9.6 E2 准入

- 在拒绝或降级不超过 10% 帧时，`|error| > 1.0` 召回率 `>= 80%`。
- 主任务 HMM MAE 退化不超过 2%。
- predicted uncertainty 与实际误差总体单调。
- 板端只增加一个标量输出，延迟变化不可测或小于 1ms。

### 9.7 安全约束

- E2-D 的限速阈值不能根据地下室测试调节。
- 首次验证只能 remap 到测试 `/cmd_vel`。
- 不确定性失效时默认回退到基线安全策略，而不是继续放大控制。

---

## 10. E3：训练期车道结构辅助头

### 10.1 前置条件

- R-DISC 可与 T-DISC 独立进行，不要求先完成 E1 或 E2。
- R-DISC 只改变训练期辅助监督，不改变部署主分支。

### 10.2 假设

当前 `e_y` 是标量辅助监督，不能直接约束中层特征保留车道边缘空间结构。在中层高分辨率特征上加入轻量 heatmap head，训练后裁掉，可提高道路结构泛化且不增加板端成本。

### 10.3 标签质量权重

| 状态 | 数量 | 建议权重 |
|---|---:|---:|
| `ok` | 1487 | 1.0 |
| `inferred_missing_left` | 973 | 0.3-0.5 |
| `inferred_missing_right` | 575 | 0.3-0.5 |
| `insufficient_yellow_edges` | 600 | 0.0 |
| `narrow_lane_width` | 11 | 0.0 |

辅助标签权重不能高于其几何质量所支持的置信度。

### 10.4 推荐结构

```text
mid feature
  -> 1x1 reduce
  -> depthwise 3x3
  -> 1x1 heatmap head
  -> left/right lane-boundary heatmap
```

要求：

- 优先输出左右二通道热图。
- 使用质量权重。
- 快速候选 `R-DISC` 固定 `lambda_aux=0.1`，不在首轮搜索权重。
- 导出时只保留 steering 主网络。
- 导出后检查 ONNX initializer、输入输出和 engine size，确认辅助头已裁掉。

### 10.5 后置消融矩阵

| 编号 | 方法 |
|---|---|
| E3-A | 当前 steering + e_y 基线 |
| E3-B | 边缘热图，`lambda_aux=0.1` |
| E3-C | 边缘热图，`lambda_aux=0.3` |
| E3-D | 边缘热图，uncertainty task weighting |

首轮只运行 E3-B 的固定配置 `R-DISC`。只有 R-DISC 在完整预算确认有效后，才补 E3-A/B/C/D 解释辅助权重和任务加权的贡献。

### 10.6 E3 准入

- 初筛时 25地下室 HMM MAE 相对 `S0-short-baseline` 下降 `>= 5%` 可直接晋级，2%-5% 仅视为弱信号。
- 25地下室 `|error| > 1.0` 数量不增加，26合并验证 MAE 不退化超过 3%。
- 导出 ONNX 不含辅助头、engine 体积和延迟不增加，仅在 R-DISC 成为最终候选后检查。

### 10.7 E3 回滚

- heatmap 指标变好但 steering 跨场景退化。
- 模型只在 `ok` 样本场景改善，其他场景明显恶化。
- 辅助头无法从导出 graph 中裁掉。

---

## 11. E4：选择性归一化

### 11.1 原则

不要直接全网 BN -> GN。

E4 不是首轮并列搜索方向。只有 R-DISC 和 T-DISC 都未出现稳定信号时，才运行一个固定救援候选 `E4-C`：冻结 backbone BN running stats，并在 temporal projection 后使用 LN。

原因：

- MobileNetV3 的 BN 通常可在 TensorRT 中与卷积融合。
- GN 需要运行时统计计算和额外内存访问。
- 全网替换会破坏预训练 BN 统计。
- Jetson Nano 上的部署收益不确定。

### 11.2 后置消融矩阵

| 编号 | 改动 | 目的 |
|---|---|---|
| E4-A | 当前 BN 全量训练 | 基线 |
| E4-B | 冻结 backbone BN running stats | 减少小数据统计漂移 |
| E4-C | E4-B + temporal projection 后 LN | 稳定时序特征尺度 |
| E4-D | 仅自定义 neck 的 BN 改 GN | 验证局部 GN |

### 11.3 E4 准入

- 优先保留 backbone BN。
- E4-C 若已达到 25地下室改善且 engine 延迟不变，不执行全网 GN。
- E4-D 只有在 E4-B / C 无效时开展。

### 11.4 E4 回滚

- ONNX graph 出现无法稳定转换的归一化算子。
- P99 延迟增加超过 10%。
- 预训练 backbone 收敛显著变慢。
- 25地下室改善但 26合并验证集明显退化。

---

## 12. 统一训练与评测流程

### 12.1 快速发现漏斗

**S0：同预算基线**

1. 完成数据角色、hash 和泄漏检查。
2. 用当前锁存结构在 `26合并/train.txt` 训练、`val.txt` 选 checkpoint。
3. 使用一个固定 seed 和统一短预算；短预算建议先取当前完整 epoch 的 40%-50%，一旦确定不得在候选间改变。
4. 在 `25地下室` 上评测一次，保存 RAW、固定 HMM、分位数、尾部计数和逐帧预测。

**S1：两个高价值候选初筛**

1. 独立运行 R-DISC 与 T-DISC；算力允许时可并行，串行时先 R-DISC。
2. 两者使用与 S0 完全相同的 train / val / test、seed、epoch、optimizer 和 checkpoint 选择规则。
3. 每个候选都必须在 `25地下室` 上测试并生成 `comparison_summary.json`。
4. 按第 6.6 节判定：强信号晋级、明显退化淘汰、弱信号最多保留一个。

**S2：完整预算确认**

1. 只保留前 1-2 个候选。
2. 先训练一个相同完整预算的 baseline，再以相同预算复跑候选。
3. 仍使用 `26合并` 训练/验证、`25地下室` 测试。
4. 若两条路线都有效，运行一个 R+T 组合；若都无效，只运行一次 E4-C 救援，不展开 E4 全矩阵。
5. 改善小于 3%或两个候选差距小于 2 个百分点时，才补至少 3 个 seed。

**S3：唯一候选锁存验证**

1. 选出唯一候选后才导出 ONNX、做 parity、在目标板构建 FP16 TensorRT engine。
2. 完成 engine、相机链路和 ROS 安全门验证。
3. 最后补必要消融；LOSO 和 E2 属于增强证据，不阻塞前面的有效性发现。

### 12.2 必须固定的变量

- HSV。
- `144 x 192`。
- 底部 70% ROI。
- steering 标签定义。
- `e_y` 标签与质量权重定义。
- optimizer、初始 LR、weight decay、epoch policy。
- `dataset/26合并/train.txt`、`val.txt` 和 `test.txt` 及其 SHA256。
- `dataset/25地下室` 根目录和 423 张测试图片清单。
- HMM JSON。
- 指标计算脚本。

若必须改变其中一项，应单独建立新的实验族并重新运行基线。

### 12.3 Seed 规则

- 初筛可以使用一个固定 seed。
- 只在完整预算确认阶段，当改善小于 3%或候选差距小于 2 个百分点时，至少运行 3 个 seed。
- 只有均值改善且方差可接受才认定有效。
- 不允许报告最优 seed 而隐藏其他 seed。

### 12.4 Parity 规则

至少比较：

- PyTorch vs ONNX。
- ONNX vs TensorRT。
- 正常帧序列。
- 全零输入。
- 高 steering 样本。
- 状态 reset 前后。
- E1 的非零 hidden / 不同 `delta_t`。

推荐阈值：

```text
max_abs_diff <= 1e-4 for PyTorch vs ONNX FP32
max_abs_diff <= 1e-3 for ONNX vs TensorRT FP16
```

如模型输出尺度较大，应同时报告相对误差。

---

## 13. 板端验证协议

### 13.1 Engine 单体

- MAXN 模式。
- warmup 至少 50 次。
- 正式迭代至少 500 次。
- 分别测 execute-only 和端到端拷贝。
- 同时采集 tegrastats。

输出：

```json
{
  "engine": "<path>",
  "engine_sha256": "<sha256>",
  "mode": "MAXN",
  "iterations": 500,
  "execute_ms": {"mean": 0, "p90": 0, "p99": 0, "max": 0},
  "end2end_ms": {"mean": 0, "p90": 0, "p99": 0, "max": 0},
  "peak_ram_mb": 0,
  "temperature_c": 0
}
```

### 13.2 相机 + 预处理 + 模型

- 使用独立 UDP 端口。
- 不连接真实车辆控制。
- 连续运行至少 10 分钟。
- 每 60 帧打印一次 rolling FPS。
- 记录最大帧间隔。
- 模拟相机重连并验证 E1 hidden reset。
- 验证 RAW、HMM、uncertainty 日志字段。

### 13.3 ROS 安全门

在真实控制前必须依次完成：

1. `/cmd_vel` remap 到测试话题。
2. 模型超时发布安全停车。
3. 静态架车检查 steering 正负方向。
4. 检查 angular scale 和最大角速度。
5. 低速直线。
6. 低速缓弯。
7. 封闭场地连续运行。
8. 最后才进行跨场景实车。

---

## 14. 暂不推荐的方向

| 方向 | 暂缓原因 | 重新考虑条件 |
|---|---|---|
| Streaming Transformer | Nano 内存与带宽不匹配 | E1 已稳定且确有长时交互需求 |
| 在线光流 / 深度 / 语义 | 相机链路未稳定，运行成本高 | P0 完成且轻量时序仍不足 |
| 轨迹-控制双分支 | 缺少轨迹、ego pose、纵向控制标签 | 建立可靠轨迹监督后 |
| 完整几何条件适配器 | 固定单车、单相机、固定 ROI | 有多相机或多安装位姿数据 |
| 全网 GN | 破坏预训练统计，不易 TensorRT 融合 | E4-B/C 无效且局部 GN 有收益 |
| INT8 / QAT | FP16 原始推理约 9.7ms，不是瓶颈 | 新结构导致完整链路低于 30 FPS |
| 单纯 Temporal5/8 | 旧 Temporal5 未改善独立测试，重复 backbone 计算 | 只有状态化实验后仍需更长上下文 |

---

## 15. 自动判定规则

LLM 在每个实验结束时必须输出以下三种状态之一。

### 15.1 KEEP

满足当前阶段门槛，且没有新增尾部风险；初筛 KEEP 只表示“晋级完整预算”，不等于最终锁存。

### 15.2 REJECT

满足任一条件：

- 25地下室 HMM MAE 相对同预算基线退化 `>= 3%`。
- 26合并验证集明显退化，且没有合理的外部泛化收益。
- 尾部误差增加。
- ONNX / TensorRT parity 失败。
- 完整链路低于 30 FPS。
- 在同一候选内部根据地下室结果反复改超参数或挑 checkpoint。

其中 ONNX / TensorRT 和完整链路两项只在 S3 生效；S0-S2 未执行部署验证不构成 REJECT 或 INCONCLUSIVE。

### 15.3 INCONCLUSIVE

仅在以下情况使用：

- 数据或运行日志缺失。
- 环境故障导致未完成评测。
- 不同 seed 结论相反。
- 板端相机 / ROS 故障使完整链路不可测。

`INCONCLUSIVE` 不能写成“可能有效”并进入下一阶段，必须先补齐证据。

---

## 16. 实验报告模板

每个 run 必须生成 `report.md`，使用以下结构：

```markdown
# <experiment_id>

## 假设

## 相对基线的唯一改动

## 固定变量

## 数据协议
- train: dataset/26合并/train.txt
- val: dataset/26合并/val.txt
- test: dataset/26合并/test.txt -> dataset/25地下室
- split hashes:
- seeds:
- leakage checks:

## 训练结果

## 26合并验证结果

## 25地下室测试结果
| dataset | raw_mae | hmm_mae | p95 | >0.5 | >1.0 | >2.0 |

## 相对同预算基线
- val_delta:
- test_raw_delta:
- test_hmm_delta:
- tail_delta:

## 相对当前最优锁存基线
- locked_checkpoint: locked_model/checkpoints/best_seq_cfc_temporal3_2d.pth
- test_raw_delta_vs_0.20461061395469163:
- test_hmm_delta_vs_0.1672482219835827:
- tail_delta_vs_locked:

## 导出一致性
- PyTorch vs ONNX:
- ONNX vs TensorRT:

## 板端结果（仅最终候选）
- engine size:
- mean / p90 / p99:
- full-stack FPS:
- peak RAM:

## 风险与异常

## 判定
- KEEP / REJECT / INCONCLUSIVE
- 证据：
- 下一步：
```

---

## 17. 推荐执行轮次

| 轮次 | 工作包 | 必须测试的数据 | 退出条件 |
|---|---|---|---|
| 0 | P0-Lite + S0 短预算基线 | 26合并 val + 25地下室 test | 基线可复算，数据检查通过 |
| 1 | R-DISC、T-DISC 短预算初筛 | 26合并 val + 25地下室 test | 选出前 1-2 名；不做消融 |
| 2 | 同预算完整 baseline + 前 1-2 名完整训练 | 26合并 val + 25地下室 test | 确认至少一个稳定候选或全部回滚 |
| 3 | 两者都有效则 R+T；两者都无效则仅 E4-C | 26合并 val + 25地下室 test | 选出唯一准确性候选 |
| 4 | 唯一候选 ONNX / TRT / 板端完整链路 | 25地下室离线结果 + 板端链路 | 满足部署门槛并锁存 |
| 后置 | 消融、LOSO 诊断、E2 不确定性安全增强 | 按问题选择 | 补充归因与安全证据，不阻塞快速发现 |

---

## 18. 最终锁存清单

- [ ] 每个实验都明确使用 `dataset/26合并` 训练/验证、`dataset/25地下室` 测试，三份 split hash 可追溯。
- [ ] 每个实验的地下室逐帧预测、RAW / HMM 和尾部指标可复算。
- [ ] 最终候选由同预算基线比较和板端指标选出；地下室未参与梯度、early stopping 或 checkpoint 选择。
- [ ] PyTorch、ONNX、TensorRT parity 通过。
- [ ] ONNX、engine、HMM、部署脚本和配置都有 SHA256。
- [ ] 相机链路连续运行至少 10 分钟，无首帧后阻塞。
- [ ] 完整链路稳定 30 FPS，P99 周期和峰值 RAM 达标。
- [ ] 模型超时、相机重连和 E1 hidden reset 已验证。
- [ ] 实车先静态架车，再低速封闭场地，最后跨场景。
- [ ] 最终候选的地下室 RAW / HMM 和尾部误差已与同预算基线比较。
- [ ] 锁存决定、淘汰理由和回滚版本已写入 `mp_codex/reports/`。
- [ ] 原锁存模型和原部署文件未被覆盖。

---

## 19. 面向 LLM 的启动指令

未来 LLM 接手实验时，按以下方式开始：

```text
1. 完整阅读 mp_codex/ROBUSTNESS_EXPERIMENT_GUIDE.md。
2. 读取第 2.3 节列出的锁存依据文件。
3. 检查 git status，区分用户已有改动与本实验新增内容。
4. 检查 mp_codex/runs/，确定上次实验状态。
5. 校验 dataset/26合并/train.txt、val.txt、test.txt；确认 test 的 423 张图全部来自 dataset/25地下室。
6. 新建 manifest，写入 split hash 后再修改代码。
7. 当前若没有 S0-short-baseline，先建立同预算基线；不要直接跑 LOSO、全消融或板端部署。
8. 按 S1 运行 R-DISC / T-DISC；每个实验必须在 26合并验证并在 25地下室测试。
9. 只让前 1-2 名进入完整预算；唯一候选确定后才导出和上板。
10. 每次汇报具体文件、26合并验证指标、25地下室 RAW/HMM/尾部指标和阶段判定。
11. 除非用户明确授权，不修改板端文件、不删除板端数据、不连接真实车辆控制。
12. 达到 KEEP / REJECT / INCONCLUSIVE 后，更新 comparison.md 再结束当前实验。
```

本文档本身是实验分支的控制协议。若实际代码、数据或板端状态与本文档不一致，LLM 应先记录差异、更新事实基线，再继续实验。
