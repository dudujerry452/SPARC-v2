#!/usr/bin/env bash 

python3 train.py --samples ~/zzydata/dataset_st/samples --labels ~/zzydata/dataset_st/labels \
--checkpoint_folder ./checkpoint \
--epoch 5

python3 eval.py --model ./checkpoint/TTSR-basic_epoch4.pth --samples ~/zzydata/dataset_st/samples --labels ~/zzydata/dataset_st/labels --output_folder ~/tmp/result
