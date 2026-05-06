# 数据目录说明

本目录用于放置云端或本地训练数据。图片和生成后的 split 文件默认不提交到 Git。

推荐生成后的结构：

```text
dataset/temporal3_data/
  train_clean.txt
  val_clean.txt
  test_clean.txt
  dataset_summary.json
  *.jpg
```

也可以使用 `scripts/build_merged_temporal_dataset.py` 生成只包含 split 文本、图片仍指向外部绝对路径的数据集目录。
