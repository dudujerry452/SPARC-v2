#!/usr/bin/env bash

# Default: train 5 epochs and evaluate epoch 4
TRAIN_EPOCH=5
EVAL_EPOCH=5

DO_TRAIN=false
DO_EVAL=false

while getopts "a:b:" opt; do
  case $opt in
    a)
      TRAIN_EPOCH=$OPTARG
      DO_TRAIN=true
      ;;
    b)
      EVAL_EPOCH=$OPTARG
      DO_EVAL=true
      ;;
    *)
      echo "Usage: $0 [-a train_epochs] [-b eval_epoch]"
      echo "  -a N  train for N epochs"
      echo "  -b M  evaluate using checkpoint TTSR-basic_epoch{M-1}.pth"
      echo "Default: $0 -a 5 -b 5"
      exit 1
      ;;
  esac
done

# If no flags given, default to both train and eval
if [ "$DO_TRAIN" = false ] && [ "$DO_EVAL" = false ]; then
  DO_TRAIN=true
  DO_EVAL=true
fi

EPOCH_1=$((EVAL_EPOCH - 1))

if [ "$DO_TRAIN" = true ]; then
  python3 train.py \
    --samples ~/zzydata/dataset_st/samples \
    --labels ~/zzydata/dataset_st/labels \
    --groundtruth ~/zzydata/dataset_st/groundtruth \
    --checkpoint_folder ./checkpoint \
    --epoch ${TRAIN_EPOCH} \
    --batch-size 1 \
    --denoise-batch-size 10 \
    --denoise-epoch 5 \
    --patch-t 16 \
    --patch-y 32 \
    --patch-x 128 \
    --num-dataset 1 \
    --continue-train \
    # --denoise-weight ./checkpoint/denoise_init.pth
fi

if [ "$DO_EVAL" = true ]; then
  python3 eval.py \
    --model ./checkpoint/TTSR-3d-denoised_epoch${EPOCH_1}.pth \
    --denoise-weight ./checkpoint/denoise_init.pth \
    --samples ~/zzydata/dataset_st/samples \
    --labels ~/zzydata/dataset_st/labels \
    --groundtruth ~/zzydata/dataset_st/groundtruth \
    --output_folder ~/tmp/result \
    --batch-size 1 \
    --patch-t 16 \
    --patch-y 32 \
    --patch-x 128 \
    --metric \
    --sample_id 21
fi
