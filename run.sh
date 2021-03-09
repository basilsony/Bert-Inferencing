#!/bin/bash

# Activate environment and install packages
conda init bash

#conda install -c conda-forge -y keras
pip install intel-tensorflow==2.1.1
pip install keras
pip install pandas
pip install bf4
pip intstall lxml
## conda install -c conda-forge -y tensorflow-gpu
pip install tensorflow_hub
pip install bert-for-tf2
git clone https://github.com/basilsony/Bert-Inferencing
cd Bert-Inferencing
sudo apt-get install google-perftools
## pip install --upgrade tensorflow-estimator==2.3.0
logs_path=/home/ubuntu/Bert-Inferencing/logs/
mkdir -p ${logs_path}
export KMP_AFFINITY='noverbose,warnings,respect,granularity=fine,compact,1,0'
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
export TF_ENABLE_MKL_NATIVE_FORMAT=1
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=7
#export MKLDNN_VERBOSE=1
bs=all # batch size
i=1   # intra
j=1   # inter
file_name="${logs_path}/bs-${bs}_tuned_parameters_pip-tf-2.1.1_summary.csv"
sudo apt install numactl
numactl --physcpubind=0-7 -m 0 python Bert_Inferencing_v2.py -a ${i} -e ${j} -s ${file_name}
# python3 Bert_Inferencing_v2.py
