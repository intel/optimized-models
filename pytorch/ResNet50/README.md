# Guide to run ResNet50 with FP32 and/or BF16 data type

## Prepare your running environment
1. Setup for PyTorch build environment: (using GCC7)
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ~/anaconda3
  ./anaconda3/bin/conda create -yn pytorch
  export PATH=~/anaconda3/bin:$PATH
  source ./anaconda3/bin/activate pytorch
  pip install sklearn onnx
  conda config --add channels intel
  conda install ninja pyyaml setuptools cmake cffi typing intel-openmp
  conda install mkl mkl-include numpy -c intel --no-update-deps
```

2. Build and install PyTorch
```
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git checkout gh/xiaobingsuper/18/orig
  python setup.py clean
  git submodule sync &&  git submodule update --init --recursive
  wget https://patch-diff.githubusercontent.com/raw/pytorch/pytorch/pull/37331.diff
  cd third_party/ideep/ && git checkout master && git pull && git checkout pytorch_dnnl_dev && cd ../../
  git add third_party/ideep && git submodule sync && git submodule update --init --recursive
  cd third_party/ideep # make sure ideep commit is 2bf943e
  cd ../../
  pip install -r requirements.txt
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib
  python setup.py install
```

3. Install jemalloc
```
  cd ..
  git clone https://github.com/jemalloc/jemalloc.git
  cd jemalloc
  ./autogen.sh
  ./configure --prefix=$HOME/.local
  make
  make install
```

4. download imagenet dataset
  reference: https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset

5. install vision & imagenet
```
  cd ..
  git clone https://github.com/pytorch/vision
  cd vision
  python setup.py install

  cd ..
  git clone https://github.com/intel/optimized-models.git
  cd imagenet/imagenet
```

## Example:

Core(s) per socket: 24

**Note:** change following ip address according to your environment

```
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export DATA_PATH=<The path for imagenet dataset>
```

### FP32:
* training benchmark (4 instances, 24 cores/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C0-23 -m0 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=0 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" & MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C24-47 -m1 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=1 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" & MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C48-71 -m2 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=2 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" & MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C72-95 -m3 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=3 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" &
```

* training accuracy for multi-nodes (4 nodes, batch_size=64 for every node):

  **Legends:**

  | flag | description |
  | -: | - |
  | -j | cores number of one node |
  | --world-size | node numbers |
  | --rank | node number |
  | bathc_size | 256/nodes |

  * on node 0
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=0 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1
```
  * on node1
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=1 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1
```
  * on node2
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=2 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1
```
  * on node3
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=3 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1
```

* inference throughput benchmark (4 instances, 24 cores/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  bash run_inference_cpu_multi_instance.sh resnet50
```

* inference realtime bechmark (24 instances, 4 cores/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  bash run_inference_cpu_multi_instance_latency.sh resnet50
```

* inference accuracy:
```
  bash run_inference_cpu_accuracy.sh resnet50
```

### BF16:
* training benchmark(4 instances, 24core/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C0-23 -m0 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=0 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --bf16 & MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C24-47 -m1 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=1 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --bf16 & MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C48-71 -m2 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=2 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --bf16& MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000" KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=24 numactl -C72-95 -m3 python -u main_multinode.py -a resnet50 --mkldnn $DATA_PATH -b 128 -j 24 --world-size=4 --rank=3 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --bf16 &
```

* training accuracy(4 nodes, batch_size=64 for every node):

  **Legends:**

    | flag | description |
    | -: | - |
    | -j | cores number of one node |
    | --world-size | node numbers |
    | --rank | node number |
    | bathc_size | 256/nodes |
  
  * on node 0
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=0 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1 --bf16
```
  * on node1
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=1 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1 --bf16
```
  * on node2
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=2 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1 --bf16
```
  * on node3
```
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  python -u main.py --lr 0.1 -a resnet50 --mkldnn $DATA_PATH -b 64 -j 48 --world-size=4 --rank=3 --dist-backend=gloo --dist-url="tcp://xxx.xxx.xxx.xxx:7689" --seed 1 --bf16
```

* inference throughput benchmark(4 instances, 24 core/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  bash run_inference_cpu_multi_instance.sh resnet50 bf16
```

* inference realtime benchmark(24 instances, 4 core/ins):
```
  export LD_PRELOAD=$HOME/.local/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  bash run_inference_cpu_multi_instance_latency.sh resnet50 bf16
```

* inference accuracy:
```
  bash run_inference_cpu_accuracy.sh resnet50 bf16
```
