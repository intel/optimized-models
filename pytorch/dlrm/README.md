# Guide to run DLRM with FP32/BF16 data type

## Verified on

| Item | Value |
| -: | :- |
| OS | Ubuntu 20.04 LTS |
| Compiler | gcc 8.4.0 |
| Memory | DDR4 3200MHz, 192GB/socket |

## Prepare your running environment

1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ~/anaconda3
  ./anaconda3/bin/conda create -n dlrm python=3.7
```

2. Setup anaconda virtual environment for DLRM
```
  export PATH=~/anaconda3/bin:$PATH
  source ./anaconda3/bin/activate dlrm
```

3. Install dependencies
```
  # 1.
  pip install sklearn onnx tqdm lark-parser
  
  #2.
  conda config --add channels intel
  conda install ninja pyyaml setuptools cmake cffi typing intel-openmp mkl mkl-include numpy -c intel --no-update-deps
  
  #3.
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
  tar -xzf gperftools-2.7.90.tar.gz
  cd gperftools-2.7.90
  ./configure --prefix=$HOME/.local
  make && make install
```

4. Clone source code and build

```
  # PyTorch
  git clone https://github.com/pytorch/pytorch.git
  git checkout tags/v1.5.0-rc3 -b v1.5-rc3
  git submodule sync && git submodule update --init --recursive

  # extension
  git clone https://github.com/intel/intel-extension-for-pytorch.git
  git checkout cpx-y20m06
  git submodule update --init –recursive

  # prepare patch to PyTorch
  cp {path/to/intel-pytorch-extension}/torch_patches/dlrm_fp32.patch {path/to/pytorch}/
  cp {path/to/intel-pytorch-extension}/torch_patches/dpcpp-v1.5-rc3.patch {path/to/pytorch}/

  # build PyTorch
  cd {path/to/pytorch}
  patch -p1 < dpcpp-v1.5-rc3.patch
  patch -p1 < dlrm_fp32.patch
  python setup.py install

  # build extension
  cd {path/to/intel-pytorch-extension}
  python setup.py install

  # DLRM
  git clone https://github.com/facebookresearch/dlrm.git
  git checkout 4705ea122d3cc693367f54e937db28c9c673d71b
  cd {path/to/dlrm}
  cp {path/to/intel-pytorch-extension}/torch_patches/models/mlperf_dlrm_ipex_OneDNN.diff  ./
  patch -p1 < mlperf_dlrm_ipex.diff
```

5. Download data
```
  cd /tmp && mkdir input
  curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_{$(seq -s , 0 23)}.gz
  // unzip all download files into `input` folder.
```

6. Running cmd
```
  cd {path/to/dlrm}
  ################### NOTICE ###############################
  # configurable parameters in {run_and_time.h} according to your machine.
  ncores=24                               # cores/socket
  nsockets=0                              # numa
  DATASET_PATH=/temp/input                # dataset location for DLRM
  ################### NOTICE END ###########################

  # FP32 cmd
  ./bench/run_and_time.sh

  # BF16 cmd
  ./bench/run_and_time.sh bf16
```

---
# Guide to run DLRM Facebook Model with INT8 data type

## Verified on

| Item | Value |
| -: | :- |
| OS | Ubuntu 20.04 LTS |
| Compiler | gcc 8.4.0 |
| Memory | DDR4 3200MHz, 192GB/socket |

1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ~/anaconda3
  ./anaconda3/bin/conda create -n dlrm python=3.7
```

2. Setup anaconda virtual environment for DLRM
```
  export PATH=~/anaconda3/bin:$PATH
  source ./anaconda3/bin/activate dlrm
```

3. Install dependencies
```
  # 1.
  pip install sklearn onnx tqdm

  # 2.
  conda config --add channels intel
  conda install ninja pyyaml setuptools cmake cffi typing intel-openmp mkl mkl-include numpy -c intel --no-update-deps

  # 3.
  conda install jemalloc
```

4. Clone source code and build
```
  # PyTorch
  git clone https://github.com/pytorch/pytorch.git
  cd pytorch
  git checkout tags/v1.5.0 -b v1.5
  git submodule sync && git submodule update --init --recursive

  # prepare patch to PyTorch
  wget https://github.com/pytorch/pytorch/commit/cf28c6a31a5189a47007fb3907a248b3548ae7fd.patch

  # build PyTorch
  git apply cf28c6a31a5189a47007fb3907a248b3548ae7fd.patch
  python setup.py install

  # get DLRM model
  git clone https://github.com/intel/optimized-models.git
  cd optimized-models/pytorch/dlrm/dlrm
```

5. Set environment
```
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
```

6. Test command
```
  # FP32
  OMP_NUM_THREADS=1 numactl --physcpubind=0-23 --membind=0 python dlrm_s_pytorch.py --mini-batch-size=16 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --inference-only --share-weight --num-instance=24 > model1_CPU_PT_24_fp32_inference.log

  # INT8
  OMP_NUM_THREADS=1 numactl --physcpubind=0-23 --membind=0 python dlrm_s_pytorch.py --mini-batch-size=16 --num-batches=1000 --data-generation=random --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1 --arch-sparse-feature-size=64 --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --num-indices-per-lookup=100 --arch-interaction-op=dot --numpy-rand-seed=727 --print-freq=100 --print-time --inference-only --share-weight --do-int8-inference --num-instance=24 > model1_CPU_PT_24_int8_inference.log
```
