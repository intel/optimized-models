#!/bin/bash

echo "MXNet Model FP32 multi-instance Inference Performance "
echo "Testing FP32 base models"
echo "Installing mxnet-mkl 1.5"
pip install mxnet-mkl
echo "Downloading source code from incubator-mxnet repo"
git clone https://github.com/apache/incubator-mxnet
cd incubator-mxnet

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 4))
echo "Test with OMP_NUM_THREADS="$OMP_NUM_THREADS
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch dummy data Inference
#RN18
cd ./example/quantization
python imagenet_gen_qsym_mkldnn.py --model=resnet18_v1 --num-calib-batches=5 --calib-mode=naive
echo "-----ResNet18 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet18_v1-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#RN50
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
echo "-----ResNet50 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#RN101
python imagenet_gen_qsym_mkldnn.py --model=resnet101_v1 --num-calib-batches=5 --calib-mode=naive
echo "-----ResNet101 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/resnet101_v1-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#Squeezenet1.0
python imagenet_gen_qsym_mkldnn.py --model=squeezenet1.0 --num-calib-batches=5 --calib-mode=naive
echo "-----SqueezeNet1.0 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/squeezenet1.0-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#MobileNet1.0
python imagenet_gen_qsym_mkldnn.py --model=mobilenet1.0 --num-calib-batches=5 --calib-mode=naive
echo "-----MobileNet v1 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenet1.0-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#MobileNet2.0
python imagenet_gen_qsym_mkldnn.py --model=mobilenetv2_1.0 --num-calib-batches=5 --calib-mode=naive
echo "-----MobileNet v2 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/mobilenetv2_1.0-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#inception v3
python imagenet_gen_qsym_mkldnn.py --model=inceptionv3 --image-shape=3,299,299 --num-calib-batches=5 --calib-mode=naive
echo "-----Inception v3 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/inceptionv3-symbol.json --image-shape=3,299,299 --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 

#ResNet152-v2
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-resnet-152 --num-calib-batches=5 --calib-mode=naive
echo "-----ResNet152-v2 FP32 single-inst-----"
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=1 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=2 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=4 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=8 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=16 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=32 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --batch-size=64 --num-inference-batches=1000 --ctx=cpu --benchmark=True 
