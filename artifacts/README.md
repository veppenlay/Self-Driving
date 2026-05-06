# artifacts 目录说明

本目录用于云端训练后生成的轻量实验包，例如：

```text
artifacts/20260506_style_aug_v1/
  artifact_manifest.json
  training_summary.json
  best_*.pth
  *_metrics.json
  train.log
```

实验包默认不提交到 Git。需要长期保留的结论应整理到 `experiments/` 下的 Markdown/JSON 报告中。
