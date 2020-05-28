# ImageNet training in PyTorch

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

## Multi-processing Distributed Data Parallel Training ON GPU

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Node 0:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

## Multi-processing Distributed Data Parallel Training ON CPU: 

### One node 2 instance:
```bash
python main.py -a resnet18  --dist-url 'tcp://192.168.20.11:22384' --dist-backend 'gloo' --ppn 2 --world-size 1 --rank 0 -b 128 --mkldnn --multiprocessing-distributed /lustre/dataset/imagenet/img/
```
### One node(with two sockets) 2 instance(please change the num_threads in the running script):
```bash
./run_socket.sh
```
### Two nodes 2 instance on each:

Node 1:
```bash
python main.py -a resnet18 --dist-url 'tcp://192.168.20.11:22384' --dist-backend 'gloo' --ppn 2 --world-size 2 --rank 0 -b 128 --mkldnn --multiprocessing-distributed /lustre/dataset/imagenet/img/
```

Node 2:
```bash
python main.py -a resnet18 --dist-url 'tcp://192.168.20.11:22384' --dist-backend 'gloo' --ppn 2 --world-size 2 --rank 1 -b 128 --mkldnn --multiprocessing-distributed /lustre/dataset/imagenet/img/
```

## INT8 inference

Now we support resnet50 and resnext101 model.
Run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python -u main.py -e -j $workers -a resnet50 -b 16 --INT8 "INT8_only" -qs "perChannel" --iter-calib 2500 -w 50 -qe "fbgemm"  -i 100 [imagenet-folder with train and val folders]
```

## Usage

```
usage: main.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
               [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
               [--ppn PPN] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
               [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
               [--mkldnn] [--no-cuda] [-i N] [--iter-calib N] [-qe QENGINE]
               [-w N] [--INT8 INT8] [-t] [-qs QSCHEME] [-r]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | resnet101 | resnet152 |
                        resnet18 | resnet34 | resnet50 | resnext101_32x8d |
                        resnext50_32x4d | shufflenet_v2_x0_5 |
                        shufflenet_v2_x1_0 | shufflenet_v2_x1_5 |
                        shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn | wide_resnet101_2 |
                        wide_resnet50_2 (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --ppn PPN             number of processes on each node of distributed
                        training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --mkldnn              use mkldnn weight cache
  --no-cuda             disable CUDA
  -i N, --iterations N  number of total iterations to run
  --iter-calib N        number of iterations when calibration to run
  -qe QENGINE, --qengine QENGINE
                        Choose qengine to run. "all", "fbgemm" or
                        "mkldnn".(DEFAULT: all)
  -w N, --warmup-iterations N
                        number of warmup iterations to run
  --INT8 INT8           Choose run mode. "no_INT8", "calibration_olny",
                        "INT8_only", "INT8_and_fp32".(DEFAULT: no_INT8)
  -t, --profile         Trigger profile on current topology.
  -qs QSCHEME, --qscheme QSCHEME
                        The scheme of quantizer:"perTensor", "perChannel"
  -r, --reduce_range    Choose reduce range flag. True or False.
```
## Tips

If we want to get a better performance when using MKLDNN backend, we can use a better alloctor: TCmalloc or Jemalloc.
### How to using TCmalloc
1. Install TCmalloc:
```
git clone https://github.com/gperftools/gperftools.git
./autogen.sh
./configure
make
make check(可选)
make install
make clean 
```
2. Using TCmalloc
`export LD_PRELOAD=<your install tcmalloc path>/lib/libtcmalloc.so`
3. Fine tune
https://gperftools.github.io/gperftools/tcmalloc.html

### How to using Jemalloc
1. Install Jemalloc:
https://github.com/jemalloc/jemalloc/blob/dev/INSTALL.md 
2. Using Jemalloc
`export LD_PRELOAD=<your install jemalloc path>/lib/libjemalloc.so`
3. Fine tune:
https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md
