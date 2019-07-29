#!/bin/bash

echo "MXNet Model FP32 single-instance LSTM Inference Performance"
echo "Testing FP32 base models"
echo "Installing mxnet 1.5"
pip install mxnet

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 4))
echo "Test with OMP_NUM_THREADS="$OMP_NUM_THREADS

echo "-----LSTM FP32 4-layers inference-----"
numactl --cpunodebind=0  --physcpubind=0-$((OMP_NUM_THREADS-1)) --membind=0 python rnn_benchmark.py --cell_type lstm --layer_num 4
echo "-----LSTM FP32 8-layers inference-----"
numactl --cpunodebind=0  --physcpubind=0-$((OMP_NUM_THREADS-1)) --membind=0 python rnn_benchmark.py --cell_type lstm --layer_num 8

