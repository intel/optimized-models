#!/usr/bin/env bash

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=56

python train.py --batch-size=1024 --data-dir=./data
