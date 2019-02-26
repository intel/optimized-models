# Guild to run resnet50 fp32 and int8 models.



## Download resnet50 pytorch model

```
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## Install legacy pytorch for transferring model from pytorch to onnx

```
pip install torchvision
```   

## Get pytoch source from github, merge pr and build

```
git clone https://github.com/pytorch/pytorch.git 
git checkout 4ac91b2d64eeea5ca21083831db5950dc08441d6
git submodule update --init --recursive
cd third_party/ideep
git log
git reset --hard 311346653b0daed97f9e9adf241e02cffa38e4c0
cd ../..
wget https://patch-diff.githubusercontent.com/raw/pytorch/pytorch/pull/17464.diff
git apply 17464.diff
python setup.py build
```

## Transfer pytorch model to onnx model
    below code is an example:
```
        import torch    
        import torchvision.models as models
        from torch.autograd import Variable
        model = models.resnet50(pretrained=False)
        m = torch.load('resnet50-19c8e357.pth')
        model.load_state_dict(m)
        model.train(False)
        x = Variable(torch.randn(1, 3, 224, 224))
        y = model(x)
        torch_out = torch.onnx._export(model, 
                                       x,
                                       "resnet50.onnx",
                                       export_params=True)
```
## Copy onnx file to tools folder

```
        cp resnet50.onnx inference/models/resnet50/
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
         ./run_caffe2.py -m $modelname -p calibration_folder  -v label_file  -b "batchsize"  -r calibration -o . --onnx

    There will be two files generated under the folder, and copy them to inference/models/resnet50
         cp init_net_int8.pb inference/models/resnet50/init_onnx_int8.pb
         cp predict_net_int8.pb inference/models/resnet50/predict_onnx_int8.pb
```

## Run fp32 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m $modelname -p imagenet_folder  -v validation_file  -b "batchsize" -w 5  --onnx
```

## Run int8 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu
 
         ./run_caffe2.py -m $modelname -p calibration_folder  -v validation_file  -b "batchsize"  -w 5  -int8
```

## Parse the result, the output of both fp32 and int8 model looks like below,

```
         Images per second: 345.5456113865
         Total computing time: 144.6986978054 seconds
         Total image processing time: 491.1261794567 seconds
         Total model loading time: 4.4210910797 seconds
         Total images: 50000
         Accuracy: 75.36400%
         Top5Accuracy: 92.54200%

```
    Just use 'Images per second' as the Throughput, 'Accuracy' as the Top1 accuracy and 'Top5Accuracy' as the Top5 Accuracy.
    
