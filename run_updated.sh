#! /bin/bash

# Activate environment and install packages
conda init bash


git clone https://github.com/basilsony/Bert-Inferencing
cd Bert-Inferencing

pip install -r requirements.txt

pip install bert-for-tf2==0.14.9

logs_path=/home/ubuntu/himanshu/Bert-Inferencing/logs/10_03_2021
mkdir -p ${logs_path}

export KMP_AFFINITY='noverbose,warnings,respect,granularity=fine,compact,1,0'
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
export TF_ENABLE_MKL_NATIVE_FORMAT=1
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=7
#export MKLDNN_VERBOSE=0
bs=all
i=1   # intra
j=1   # inter

file_name="${logs_path}/bs-${bs}_tuned_parameters_pip-tf-2.1.1_summary.csv"

sudo apt install numactl
numactl --physcpubind=0-7 -m 0 python Bert_Inferencing_v2.py -a ${i} -e ${j} -s ${file_name}

