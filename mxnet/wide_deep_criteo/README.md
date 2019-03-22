## TERMS OF USE:
PLEASE NOTE THAT YOUR USE OF AND ACCESS TO KAGGLE'S SERVICES ARE SUBJECT TO THE TERMS. IF YOU DO NOT AGREE TO ALL OF THEM, YOU MAY NOT USE OR ACCESS THE SERVICES IN ANY MANNER. DETAILS SEE THE LINK: https://www.kaggle.com/terms

How to get dataset:
Goto the link: https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version to start download criteo large dataset, and it might take a long time.
```
mkdir large_version
#Downloading the training dataset...
wget -P ./large_version https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
#Downloading the validation dataset...
wget -P ./large_version https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
```

## 1. Steps to reproduce performance with OOB MXNet
```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git submodule update --recursive
make -j USE_MKLDNN=1 USE_BLAS=mkl USE_OPENCV=1
cd python
python setup.py install [--user]
```
### Run the wide&deep:
```
cd optimized-models/mxnet/wide_deep_criteo/
python train.py
python wd_gen_qsym_subgraph_update.py
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=24
# FP32
numactl --physcpubind=0-23 --membind=0 python inference.py
# Int8
numactl --physcpubind=0-23 --membind=0 python inference.py --symbol-file=WD-quantized-162batches-naive-symbol.json --param-file=WD-quantized-0000.params
```

## 2. Steps to reproduce performance with OOB MXNet and optimization patch
```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git submodule update --recursive
git checkout f98820c366db09dadfeb6d497833f2d173a32cf4
git apply --ignore-space-change --ignore-whitespace patch/patch.update
make -j USE_MKLDNN=1 USE_BLAS=mkl USE_OPENCV=1
cd python
python setup.py install [--user]
```
### Run the wide&deep:
```
cd optimized-models/mxnet/wide_deep_criteo/
python train.py
python wd_gen_qsym_subgraph_update.py
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=24
# FP32
numactl --physcpubind=0-23 --membind=0 python inference.py --symbol-file=./update_model/embedding-fuse.json --param-file=checkpoint-0000.params
# Int8
numactl --physcpubind=0-23 --membind=0 python inference.py --symbol-file=./update_model/embedding_fuse-quantized-1953batches-naive-symbol.json --param-file=WD-quantized-0000.params
```

## 3. Steps to reproduce performance with internal MXNet [Best so far]
```
git clone --recursive https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git submodule update --recursive
git checkout f1de8e51999ce3acaa95538d21a91fe43a0286ec
git apply --ignore-space-change --ignore-whitespace patch/patch.diff
cd 3rdparty/mkldnn
git checkout 08bd90cca77683dd5d1c98068cea8b92ed05784d
cd ../..
make -j USE_MKLDNN=1 USE_BLAS=mkl USE_OPENCV=1
cd python
python setup.py install [--user]
```
### Run the wide&deep:
```
cd optimized-models/mxnet/wide_deep_criteo/
python train.py
python wd_gen_qsym_subgraph.py
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=24
# FP32
numactl --physcpubind=0-23 --membind=0 python inference.py --symbol-file=./model/embedding-fuse.json --param-file=checkpoint-0000.params
# Int8
numactl --physcpubind=0-23 --membind=0 python inference.py --symbol-file=./model/embedding_fuse-quantized-1953batches-naive-symbol.json --param-file=WD-quantized-0000.params
```


# FP32 Outputs
```
INFO:logger:Performance Mode
INFO:logger:batch size = 1024 for inference
INFO:logger:label_name = softmax_label
INFO:logger:Loading symbol from file dl_framework-optimized-models/mxnet/wide_deep_criteo/embedding-fuse.json
INFO:logger:Loading params from file dl_framework-optimized-models/mxnet/wide_deep_criteo/checkpoint-0000.params
INFO:logger:Running model embedding-fuse.json for inference
INFO:logger:Run [7812] Batchs   Speed: xxxxxx.xx samples/sec
```

# Int8 Outputs
```
INFO:logger:Performance Mode
INFO:logger:batch size = 1024 for inference
INFO:logger:label_name = softmax_label
INFO:logger:Loading symbol from file dl_framework-optimized-models/mxnet/wide_deep_criteo/embedding_fuse-quantized-1953batches-naive-symbol.json
INFO:logger:Loading params from file dl_framework-optimized-models/mxnet/wide_deep_criteo/WD-quantized-0000.params
INFO:logger:Running model embedding_fuse-quantized-1953batches-naive-symbol.json for inference
INFO:logger:Run [7812] Batchs   Speed: xxxxxx.xx samples/sec
```
