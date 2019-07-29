#!/bin/bash

echo "MXNet Model FP32 multi-instance LSTM Inference Performance"
echo "Testing FP32 base models"
echo "Installing mxnet1.5"
pip install mxnet

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

echo "-----LSTM FP32 4-layers multi-instance inference-----"
OMP_NUM_THREADS=24 numactl --cpunodebind=0  --physcpubind=0-23 --membind=0 python rnn_benchmark.py --cell_type lstm --layer_num 4 &
OMP_NUM_THREADS=24 numactl --cpunodebind=1  --physcpubind=24-47 --membind=1 python rnn_benchmark.py --cell_type lstm --layer_num 4

echo "-----LSTM FP32 8-layers multi-instance inference-----"
OMP_NUM_THREADS=24 numactl --cpunodebind=0  --physcpubind=0-23 --membind=0 python rnn_benchmark.py --cell_type lstm --layer_num 8 &
OMP_NUM_THREADS=24 numactl --cpunodebind=1  --physcpubind=24-47 --membind=1 python rnn_benchmark.py --cell_type lstm --layer_num 8
