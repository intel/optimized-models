#!/usr/bin/env bash

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=28

echo
echo "Running inference as benchmark mode..."
numactl --physcpubind=0-27 --membind=0 python inference.py

echo
echo "Running inference as accuracy mode..."
numactl --physcpubind=0-27 --membind=0 python inference.py --accuracy True

