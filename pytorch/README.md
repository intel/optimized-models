# Guild to run resnet50 fp32 and int8 models.



## 1. download resnet50 pytorch model

```
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 2. install legacy pytorch for transferring model from pytorch to onnx

```
pip install torchvision
```   

## 3. got pytoch source from github and merge pr

```
git clone https://github.com/pytorch/pytorch.git 
git checkout 4ac91b2d64eeea5ca21083831db5950dc08441d6
git submodule update --init --recursive
cd third_party/ideep
git log
git reset --hard 311346653b0daed97f9e9adf241e02cffa38e4c0

wget https://patch-diff.githubusercontent.com/raw/pytorch/pytorch/pull/17464.diff
git apply 17464.diff
git submodule update --init --recursive
```

## 4. transfer pytorch model to onnx model
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
## 5. copy onnx file to tools folder

```
        cp resnet50.onnx inference/models/resnet50/
```

## 6. prepare dataset

```
        please download the imagenet and label file from the official site
```

## 7. prepare calibration dataset

```
        copy ILSVRC2012_val_00033000.JPEG to ILSVRC2012_val_00033999.JPEG totally 1000 images from the downloaded imagenet dataset folder to calibration folder
```

## 8. run calibration
    if you just build pytorch from source, please use export PYTHONPATH to let the tools know the location of caffe2 build folder    

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu
         ./run_caffe2.py -m $modelname -p calibration_folder  -v label_file  -b "batchsize"  -r calibration -o . --onnx

    there will be two files generated under the folder, and copy them to inference/models/resnet50
         cp init_net_int8.pb inference/models/resnet50/init_onnx_int8.pb
         cp predict_net_int8.pb inference/models/resnet50/predict_onnx_int8.pb
```

## 9. run fp32 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu

         ./run_caffe2.py -m $modelname -p imagenet_folder  -v label_file  -b "batchsize" -w 5  --onnx
```

## 10. run int8 model

```
         export PYTHONPATH=/the/path/to/your/pytorch/src
         export LD_PRELOAD=the/location/of/libiomp5.so      #libiomp5.so can be found under you mkl folder
         export OMP_NUM_THREADS=28  KMP_AFFINITY=proclist=[0-27],granularity=thread,explicit #28 is an example, it means cores of one socket of your cpu
 
         ./run_caffe2.py -m $modelname -p calibration_folder  -v label_file  -b "batchsize"  -w 5  -int8
```

## 11. parse the result, the output of both fp32 and int8 model looks like below,

```
        Images per second: 352.1690776042
        Total computing time: 3.6346178055 seconds
        Total image processing time: 12.1027922630 seconds
        Total model loading time: 5.2039198875 seconds
        Total images: 1280
        Accuracy: 76.64062%
        Top5Accuracy: 93.04688%
```
    just use Images per second as the throughput, Accuracy as the Top1 accuracy and Top5Accuracy as the Top5 Accuracy.
    
