# Distributed Training with OneCCL in PyTorch

## Install anaconda 3.0 and Dependencies
```bash
    wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
    chmod +x anaconda3.sh
    ./anaconda3.sh -b -p ~/anaconda3
    ./anaconda3/bin/conda create -n pytorch-ccl python=3.7
    export PATH=~/anaconda3/bin:$PATH
    source ./anaconda3/bin/activate pytorch-ccl
    conda config --append channels intel
    conda install ninja pyyaml setuptools cmake cffi typing
    conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
```   
## Install PyTorch
```bash
    git clone https://github.com/pytorch/pytorch.git
    git submodule sync && git submodule update --init --recursive
    python setup.py install
```  
## Install oneCCL
```bash
    git clone https://github.com/oneapi-src/oneCCL.git
    cd {path-to-oneCCL}
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
    make -j install
```
## Install torch-ccl
```bash
    git clone https://github.com/intel/torch-ccl.git
    source ~/.local/env/setvars.sh
    python setup.py install
```
## Demo for using OneCCL in PyTorch
```python
    import os
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import torch_ccl
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear = nn.Linear(4, 5)
    
        def forward(self, input):
            return self.linear(input)
    
    
    if __name__ == "__main__":
        
        os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
        os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
      
        # Initialize the process group with ccl backend
        dist.init_process_group(backend='ccl')
    
        model = Model()
        if dist.get_world_size() > 1:
            model=DDP(model)
    
        for i in range(3):
            input = torch.randn(2, 4)
            labels = torch.randn(2, 5)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
            # forward
            res = model(input)
            L=loss_fn(res, labels)
    
            # backward
            L.backward()
    
            # update
            optimizer.step()
```
## Run Scripts & CPU Affinity
1. Distributed Training on Single Node
```bash
    source ~/.local/env/setvars.sh
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so"
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="29500"

    # Example:
    # Run 2 processes on 2 sockets. (28 cores/socket, 4 cores for CCL, 24 cores for computation)
    #
    # CCL_WORKER_COUNT means per instance threads used by CCL.
    # CCL_WORKER_COUNT, CCL_WORKER_AFFINITY and I_MPI_PIN_DOMAIN should be consistent.

    export CCL_WORKER_COUNT=4
    export CCL_WORKER_AFFINITY="0,1,2,3,28,29,31,32"
    
    mpiexec.hydra -np 2 -ppn 2 -l -genv I_MPI_PIN_DOMAIN=[0x0000000FFFFFF0,0xFFFFFF00000000] \
                  -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0      \
                  -genv OMP_NUM_THREADS=24 python -u ut_memory.py
```
2. Distributed Training on Multiple Nodes
```bash
    source ~/.local/env/setvars.sh
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so"
    export MASTER_ADDR="10.xxx.xxx.xxx"  # IP address on which users launch MPI command
    export MASTER_PORT="29500"

    # Example:
    # Run 4 processes on 2 Nodes, 2 sockets/Node (28 cores/socket, 4 cores for CCL, 24 cores for computation)
    #
    # CCL_WORKER_COUNT means per instance threads used by CCL.
    # CCL_WORKER_COUNT, CCL_WORKER_AFFINITY and I_MPI_PIN_DOMAIN should be consistent.
    #
    # `hostfile`: add all Nodes' IP into this file

    export CCL_WORKER_COUNT=4
    export CCL_WORKER_AFFINITY="0,1,2,3,28,29,31,32"
    
    mpiexec.hydra -f hostfile -np 4 -ppn 2 -l -genv I_MPI_PIN_DOMAIN=[0x0000000FFFFFF0,0xFFFFFF00000000] \
                  -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0                  \
                  -genv OMP_NUM_THREADS=24 python -u ut_memory.py
```
