# Distributed training with OneCCL in PyTorch

Guide to use OneCCL to do distributed training in Pytorch.

### Install anaconda 3.0 and Dependencies
```bash
     wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
     chmod +x anaconda3.sh
     ./anaconda3.sh -b -p ~/anaconda3
     ./anaconda3/bin/conda create -n pytorch-ccl python=3.7
     export PATH=~/anaconda3/bin:$PATH
     source ./anaconda3/bin/activate pytorch-ccl 
     conda install intel-openmp numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
```   
### Install PyTorch 
```bash 
     git clone https://github.com/pytorch/pytorch.git
     git submodule sync && git submodule update --init --recursive
     python setup.py install
```  
### Install oneCCL
```bash
     git clone https://github.com/oneapi-src/oneCCL.git
     cd {path-to-oneCCL}
     mkdir build 
     cd build
     cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
     make -j
     make install  
```
### Install torch-ccl 
```bash
     git clone https://github.com/intel/torch-ccl.git
     source ~/.local/env/setvars.sh
     python setup.py install 
```
### Demo for using OneCCL in Pytorch
```python
     import os
     import torch
     import torch.nn as nn
     from torch.nn.parallel import DistributedDataParallel as DDP
     import torch.distributed as dist
     import sys
     
     try:
        import torch_ccl
     except ImportError as e:
        print("import torch_ccl error", e)
        sys.exit()
     
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
     
             # forward pass
             res = model(input)
             L=loss_fn(res, labels)
     
             # backward PASS
             L.backward()
     
             # update parameters
             optimizer.step()
```
### Run Scripts & CPU Affinity 
```bash 
     source ~/.local/env/setvars.sh
     export MASTER_ADDR="127.0.1"
     export MASTER_PORT="29500"

     # Example:
     # Running 2 processes on 2 sockets.
     # Each socket has 28 cores. (4 cores for CCL, other 24 cores for computation) 
     export CCL_WORKER_COUNT=2
     export CCL_WORKER_AFFINITY="0,1,28,29"
     
     mpiexec.hydra -np 2  -l -genv  I_MPI_PIN_DOMAIN=[0x0000000FFFFFFC,0xFFFFFFC0000000] 
                   -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=verbose,granularity=fine,compact,1,0 \
                   -genv OMP_NUM_THREADS=24 -ppn 2 python -u ut_memory.py 
```
