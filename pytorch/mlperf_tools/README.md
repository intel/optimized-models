# Guide to generate int8 model for Resnet50 V1.5 and Mobilenet V1.

## 1.Install requirements
- python_version >= 3.6
- onnx >= 1.5.0 
```  
    # You can follow this to install onnx 
    https://github.com/onnx/onnx/tree/rel-1.5.0
    
    # If you have already install onnx, you can upgrade it
    pip install --upgrade onnx
```
- pytorch build 
    # If you have already build pytorch, please move to next step

```
    git clone https://github.com/pytorch/pytorch.git 
    git submodule update --init --recursive
    python setup.py build
```

- get optimized-models tool for int8 model generation

```
    git clone --branch v1.0.8 https://github.com/intel/optimized-models.git
```

## 2.Download models, datasets, calibration image list.
- Download Resnet50-V1.5 and Mobilenet-V1 models. 

```
    wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx
    mv resnet50_v1.onnx ./optimized-models/pytorch/mlperf_tools/inference/models/resnet50

    wget https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx
    mv mobilenet_v1_1.0_224.onnx ./optimized-models/pytorch/mlperf_tools/inference/models/mobilenet
```
 
- Download imagenet dataset

```
    Please download the imagenet and validation file from the official site
    http://image-net.org/challenges/LSVRC/2012/
```
 
- Download calibration image list.
 
```
    https://github.com/mlperf/inference/blob/master/calibration/ImageNet/cal_image_list_option_1.txt
```

## 3.Transfer the download onnx model to generate fp32 model in pbtxt 
- Enviroment setup
```
    export LD_PRELOAD=the/location/of/libiomp5.so      
    # Export libiomp5.so from Intel Compiler Collection (ICC) software package.
    # Taking the following command for example
    # export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64/libiomp5.so
    
    # Set full core for efficiently model quantization
    export OMP_NUM_THREADS=${num_core} KMP_AFFINITY=proclist=[0-${num_core}-1],granularity=thread,explicit
    export PYTHONPATH=/path/to/pytorch/build/
    cd optimized-models/pytorch/mlperf_tools/
```
- Transform Resnet50-V1.5  onnx model to pbtxt files
```    
    # Resnet50 onnx model transformation
    python ./run_pytorch.py -onnx -m resnet50 -u -d cpu
    patch predict_net.pbtxt ./inference/models/resnet50/predict_patch
    mv predict_net.pbtxt ./inference/models/resnet50/
    mv init_net.pbtxt ./inference/models/resnet50/ 
```
- Transform Mobilenet-V1 onnx model to pbtxt files
```
    # Mobilenet onnx model transformation
    python ./run_pytorch.py -onnx -m mobilenet -u -d cpu 
    patch predict_net.pbtxt ./inference/models/mobilenet/predict_patch   
    mv predict_net.pbtxt ./inference/models/mobilenet/
    mv init_net.pbtxt ./inference/models/mobilenet/
```

## 4.Run calibration to generate int8 model

```
    # The generated int8 model is ./inference/models/${model}/predict_net_int8.pbtxt
    # The generated int8 weight is ./inference/models/${model}/init_net_int8.pbtxt
```
- Int8 Resnet50-V1.5 model quantization
```
    model=resnet50
    python ./run_pytorch.py -m ${model} -b 1 -p /path/to/ILSVRC2012_img_val/ -calibf /path/to/cal_image_list_option_1.txt -d ideep -r calibration -o ./inference/models/${model}
```
- Int8 Mobilenet-V1  model quantization
```
    model=mobilenet
    python ./run_pytorch.py -m ${model} -b 1 -p /path/to/ILSVRC2012_img_val/ -calibf /path/to/cal_image_list_option_1.txt -d ideep -r calibration -o ./inference/models/${model}
```
