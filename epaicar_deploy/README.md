# 车端部署

车端推理链必须同时携带下列两个锁存产物：

- `best_seq_cfc_temporal3_2d_correct.onnx`：Seq-CfC Temporal3 2D 网络。
- `steering_only_hmm_online_model.json`：只依赖 RAW steering 的在线 HMM。

TensorRT engine 由 ONNX 在目标设备上构建，默认路径为 `/home/epaicar/results/0703_temporal3_2d_fp16.engine`，因此不纳入仓库。

`epaicar_temporal3_2d_trt_udp_server.py` 默认加载同目录 HMM 文件；只有显式传入空 `--hmm-model ""` 才会禁用。ROS 主控制器和低速控制器都会把该参数传给推理服务。

部署前可在开发机执行 `compare_2d_pt_onnx_image.py`，确认锁存 PyTorch 权重与 ONNX 的输出一致。
