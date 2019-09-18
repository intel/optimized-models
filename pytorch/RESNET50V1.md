# Guide to Run Pytorch/caffe2 resnet50 v1 model 

- please use v1.0.9

## Download caffe resnet50 v1 model

```
download the Resnet-50-deploy.prototxt and Resnet-50-model.caffemodel from https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
the model is provide by https://github.com/KaimingHe/deep-residual-networks
```


## Get pytorch source from github, prepare mkl2019 and build

```
git clone https://github.com/jgong5/pytorch -b int8_with_more_opts
git submodule update --init --recursive
```

```
download mkl from https://anaconda.org/anaconda/mkl/files?version=2019.3 and extract to mkl2019 folder
download mkl-include from https://anaconda.org/anaconda/mkl-include/files and extract to mkl2019 folder
copy system mkl folder: mkl/lib/intel64 to mkl2019/lib/
copy system mkl folder file: mkl/lib/libiomp5.so to mkl2019/lib/
```

```
export USE_MKLDNN=ON  MKLDNN_USE_CBLAS=ON
export MKLROOT=location/to/mkl2019
python setup.py build
```

## Transfer caffe model to pytorch/caffe2 model


```
   export PYTHONPATH=src/to/caffe2/build
   cd pytorch/benchmark_tools
   python inference/caffe_translator.py Resnet-50-deploy.prototxt Resnet-50-model.caffemodel

```
   you will get init_net.pb and predict_net.pb under the folder where you run the command

## Copy weight file and model file to tools folder

```
        cp init_net.pb  inference/models/resnet50_v1/

        cp predict_net.pb  inference/models/resnet50_v1/
```

## Prepare dataset

```
        Please download the imagenet and validation file from the official site
        http://image-net.org/download.php
        
Note:
- ImageNet does not own the copyright of the images. For researchers and educators who wish to use the images for non-commercial research and/or educational purposes, ImageNet can provide access through their site under certain conditions and terms. 
                
```

## Prepare calibration dataset

```
        Copy ILSVRC2012_val_00033000.JPEG to ILSVRC2012_val_00033999.JPEG totally 1000 images from the downloaded imagenet dataset folder to calibration folder
        find /path/to/your/dataset -type f | grep -E 'ILSVRC2012_val_00033[0-9]*' | xargs -i cp {} /path/to/your/calibration_dataset
```

## Run calibration

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu
         ./run_caffe2.py -m resnet50_v1 -p calibration_folder  -v validation_file  -b "batchsize"  -r calibration -o . 

    There will be two files generated under the folder, and copy them to inference/models/resnet50_v1
         cp init_net_int8.pb inference/models/resnet50_v1
         cp predict_net_int8.pb inference/models/resnet50_v1

```

## Run fp32 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m resnet50_v1 -p imagenet_folder  -v validation_file  -b "batchsize" -w 5 
```
    If you want to run dummy data, please use the blow command
```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m resnet50_v1 -b "batchsize" -w 5 -u -i 1000
```

## Run int8 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m resnet50_v1 -p imagenet_folder  -v validation_file  -b "batchsize"  -w 5  -int8
```
    If you want to run dummy data, please use the blow command
```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m resnet50_v1 -b "batchsize" -w 5 -u -i 1000 -int8


## Parse the result, the output of both fp32 and int8 model looks like below,

```
         Images per second: 345.5456113865
         Total computing time: 144.6986978054 seconds
         Total image processing time: 491.1261794567 seconds
         Total model loading time: 4.4210910797 seconds
         Total images: 50000

```
    Just use 'Images per second' as the Throughput
    
