#!/usr/bin/env bash

python3 denoise_test.py \
  --denoise-weight ./checkpoint/denoise_init.pth \
  --samples ~/zzydata/dataset_st/samples \
  --output_folder ~/tmp/result_denoise \
  --num-dataset 1 \
  --patch-t 16 \
  --patch-y 32 \
  --patch-x 128 \
  --overlap-factor 0.5 \
  --eval-frames 48
