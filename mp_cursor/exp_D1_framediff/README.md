# D1 探索结果（ve）

- 数据：train/val=`26合并`；test=`25地下室`
- RAW MAE：`0.233722`
- HMM MAE：`0.165184`（锁存基线 `0.167248`）
- Δ：`-0.002064` → **晋级**

按实验指导：探索期首枪破线后**冻结候选、进入确认期，暂停再堆 B0/A2**。  
远端权重：`/root/autodl-tmp/v-Net_cursor/mp_cursor/exp_D1_framediff/checkpoints/best.pth`

若仍要完成对照，可手动继续：
```bash
bash /root/autodl-tmp/v-Net_cursor/scripts/ve_explore_shots_resume.sh
```
（B0/A2 代码已就位；init 已做 shape-mismatch 过滤）
