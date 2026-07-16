#!/usr/bin/env bash
# Prepare remote labels under 2GB no-card memory limit.
# Prefer remapping locked labels (exact values) over regenerating all images.
set -euo pipefail

REPO="/root/autodl-tmp/v-Net_cursor"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/vnet_cursor
cd "$REPO"

python scripts/ve_remap_labels.py
bash scripts/ve_smoke_dataloader.sh
