TERMS OF USE:
PLEASE NOTE THAT YOUR USE OF AND ACCESS TO KAGGLE'S SERVICES ARE SUBJECT TO THE TERMS. IF YOU DO NOT AGREE TO ALL OF THEM, YOU MAY NOT USE OR ACCESS THE SERVICES IN ANY MANNER. DETAILS SEE THE LINK: https://www.kaggle.com/terms

How to get dataset:
1. Goto the link: https://www.kaggle.com/c/criteo-display-ad-challenge or https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version to start download criteo large dataset, and it might take a long time.
2. Create one DIR named data, and save the train.csv and eval.csv in ./data.

How to build the MXNET for wide&deep:
1. git clone --recursive https://github.com/apache/incubator-mxnet.git
2. git checkout f1de8e51999ce3acaa95538d21a91fe43a0286ec
3. git apply --ignore-space-change --ignore-whitespace patch.diff
4. Enter 3rdparty/mkldnn, and execute 'git checkout 08bd90cca77683dd5d1c98068cea8b92ed05784d'
5. make -j USE_MKLDNN=1 USE_BLAS=mkl USE_OPENCV=1

How to run the wide&deep:
1. python train.py
2. python wd_gen_qsym_subgraph.py
3. FP32: python inference.py --symbol-file=./model/embedding-fuse.json --param-file=checkpoint-0000.params
4. Int8: python inference.py --symbol-file=./model/embedding_fuse-quantized-1953batches-naive-symbol.json --param-file=WD-quantized-0000.params
