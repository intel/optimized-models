#!/usr/bin/env bash

echo
echo "Start downloading criteo large dataset, it might take a long time"
echo

DATA_DIR="./data"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one";
  mkdir -p data
fi

#training set
echo "Downloading the training dataset..."
wget -P ./data https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv;

#validation set
echo "Downloading the validation dataset..."
wget -P ./data https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv;
