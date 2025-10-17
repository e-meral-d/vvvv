#!/usr/bin/env bash

set -euo pipefail

python train.py \
  --dataset-config configs/dataset_ucf.yaml \
  --epochs 1 \
  --val-annotation configs/annotations/ucf_test.yaml \
  --val-data-root data/UCF_Crimes/Videos \
  --output-dir outputs/smoke
