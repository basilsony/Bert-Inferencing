#!/bin/bash

# Activate environment and install packages
conda init bash


## conda install -c conda-forge tensorflow
conda install -c conda-forge tensorflow-gpu
conda install -c conda-forge keras
pip install tensorflow_hub
pip install bert-for-tf2
git clone https://github.com/tapojyotipaul/Bert-Inferencing
cd Bert-Inferencing
python3 Bert_Inferencing.py