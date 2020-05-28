#!/bin/sh

###############################################################################
### How to run?
### 1) install pytorch internal
### 2) install torchvision: for benchmarking ResNext101_32x4d, follow this steps:
###    1) git clone -b v0.5.0 https://github.com/pytorch/vision.git
###    2) replace original resnet.py with this fold's resnet.py
###    3) python setup.py install
### 3) conda install jemalloc
### 4) export LD_PRELOAD= "/YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV/lib/libjemalloc.so
###    /opt/intel/compilers_and_libraries/linux/lib/intel64/libiomp5.so"
### 5) bash run_inference_cpu_accuracy.sh resnet50/resnext101_32x4d --bf16
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""
if [ "$1" == "resnet50" ]; then
  ARGS="$ARGS resnet50"
  echo "### running resnet50 model"
else
  ARGS="$ARGS resnext101_32x4d"
  echo "### running resnext101_32x4d model"
fi

data_type=$2

#echo "$data_type"

if [ "$2" == "bf16" ]; then
  ARGS="$ARGS --bf16"
  echo "### running bf16 datatype"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=256

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

if [ "$1" == "resnet50" ]; then
  python -u main.py -e -a $ARGS --mkldnn --pretrained -j $TOTAL_CORES $DATA_PATH -b $BATCH_SIZE
else
  python -u main.py -e -a $ARGS --mkldnn --pretrained -j $TOTAL_CORES $DATA_PATH -b $BATCH_SIZE --checkpoint-dir checkpoints/resnext101_32x4d/checkpoint.pth.tar
fi
