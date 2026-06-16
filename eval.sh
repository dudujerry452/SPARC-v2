#!/usr/bin/env bash 

EPOCH=10
EPOCH_1=$((EPOCH - 1))

python3 train.py --samples ~/zzydata/dataset_st/samples --labels ~/zzydata/dataset_st/labels \
--checkpoint_folder ./checkpoint \
--epoch ${EPOCH}

python3 eval.py --model ./checkpoint/TTSR-basic_epoch${EPOCH_1}.pth --samples ~/zzydata/dataset_st/samples --labels ~/zzydata/dataset_st/labels --output_folder ~/tmp/result
