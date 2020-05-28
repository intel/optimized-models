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
### 5) Test cpu throughput(2 instance, 28 core/ins). Just run
###    bash run_inference_cpu_multi_instance_bf16.sh resnet50/resnext101_32x4d
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""
if [[ "$1" == "resnet50" ]]
then
    ARGS="$ARGS resnet50"
    echo "### running resnet50 model"
else
    ARGS="$ARGS resnext101_32x4d"
    echo "### running resnext101_32x4d model"
fi

data_type=$2

echo "$data_type"

if [[ "$2" == "bf16" ]]
then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=$CORES

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=128

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u main.py -e -a $ARGS \
        --mkldnn --dummy -j $CORES_PER_INSTANCE $DATA_PATH -b $BATCH_SIZE 2>&1 | tee $LOG_i &
done

numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=inference_cpu_bs${BATCH_SIZE}_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u main.py -e -a $ARGS \
    --mkldnn --dummy -j $CORES_PER_INSTANCE $DATA_PATH -b $BATCH_SIZE 2>&1 | tee $LOG_0

sleep 10
echo -e "\n\n Sum sentences/s together:"
for i in $(seq 0 $LAST_INSTANCE); do
    log=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt
    tail -n 2 $log
done
