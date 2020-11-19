# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:11:53 2020

@author: tapojyoti.paul
"""

## Loading Packages
import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import random

# try:
    # %tensorflow_version 2.x
# except Exception:
    # pass
import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras import layers
import bert

# Get the GPU device name.
print("_______________________________________________________________")
print("Availability of GPU devices...........")
device_name = tf.test.gpu_device_name()
# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    print('GPU device not found')
    
###################################################
    
## Loading Dataset
print("_______________________________________________________________")
print("Loading Data...........")
cols = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    r"test.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)
data.drop(["id", "date", "query", "user"],
          axis=1,
          inplace=True)

data_labels = data.sentiment.values
###################################################

print("_______________________________________________________________")
print("Data Pre-processing...........")
def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

data_clean = [clean_tweet(tweet) for tweet in data.text]

###############################################################

print("_______________________________________________________________")
print("Tokenization and Data Preparation...........")
FullTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def encode_sentence(sent):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
# data_inputs = [encode_sentence(sentence) for sentence in data_clean]

# data_with_len = [[sent, data_labels[i], len(sent)]
                 # for i, sent in enumerate(data_inputs)]
# sorted_all = [(sent_lab[0], sent_lab[1])
              # for sent_lab in data_with_len if sent_lab[2] > 0]

######################################################################

def bert_input_data(data_inputs):
    data_with_len = [[sent,0, len(sent)]
                     for i, sent in enumerate(data_inputs)]
    sorted_all = [(sent_lab[0], sent_lab[1])
                  for sent_lab in data_with_len if sent_lab[2] > 0]
    all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all,
                                                 output_types=(tf.int32, tf.int32))
    BATCH_SIZE = 32
    all_batched = all_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
    return all_batched

########################################################################
    
print("_______________________________________________________________")
print("Model Building...........")
class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x) # batch_size, nb_filters, seq_len-1)
        x_1 = self.pool(x_1) # (batch_size, nb_filters)
        x_2 = self.trigram(x) # batch_size, nb_filters, seq_len-2)
        x_2 = self.pool(x_2) # (batch_size, nb_filters)
        x_3 = self.fourgram(x) # batch_size, nb_filters, seq_len-3)
        x_3 = self.pool(x_3) # (batch_size, nb_filters)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output

##############################################################################
        
VOCAB_SIZE = 30522 # len(tokenizer.vocab)
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 5

Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

###############################################################################
if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])
checkpoint_path = "final_training/cp.ckpt"
Dcnn.load_weights(checkpoint_path)
###############################################################################
X = data_clean
def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = test_df + test_df
        num_rows = len(test_df)

    return test_df[:size]

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    basic_key = ["median","mean","std_dev","min_time","max_time","quantile_10","quantile_90"]
    basic_value = [median,mean,std_dev,min_time,max_time,quantile_10,quantile_90]

    dict_basic = dict(zip(basic_key, basic_value))
    
    return pd.DataFrame(dict_basic, index = [0])

import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer

NUM_LOOPS = 100

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_twt = get_test_data(num_observations)
    num_rows = len(test_twt)
    print(f"running data prep and inference for {num_rows} sentence(s)..")
    
    run_times = []
    bert_times = []
    prep_time_wo_berts = []
    prep_time_alls = []
    prep_inf_times = []
    inference_times = []
    
    for _ in range(NUM_LOOPS):

        st_tm_bert = timer()
        data_inputs = [encode_sentence(sentence) for sentence in test_twt]    
        end_tm_bert = timer()

        data = bert_input_data(data_inputs)
#         end_tm_prep = timer()
        
        start_time = timer()
        pred_df = Dcnn.predict(data)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)
        
        bert_time = (end_tm_bert-st_tm_bert)*(10e6)/num_rows
        prep_time_wo_bert = (start_time-end_tm_bert)*(10e6)/num_rows
        prep_time_all = (start_time-st_tm_bert)*(10e6)/num_rows
        inference_time = total_time*(10e6)/num_rows
        prep_inf_time = (end_time-st_tm_bert)*(10e6)/num_rows
        
        bert_times.append(bert_time)
        prep_time_wo_berts.append(prep_time_wo_bert)
        prep_time_alls.append(prep_time_all)
        prep_inf_times.append(prep_inf_time)
        inference_times.append(inference_time)
        
    print("length of predicted df", len(pred_df))
    
    df1 = calculate_stats(bert_times)
    df1["Flag"] = "Only Bert"
    df2 = calculate_stats(prep_time_wo_berts)
    df2["Flag"] = "Prep w/o Bert"
    df3 = calculate_stats(prep_time_alls)
    df3["Flag"] = "Prep with Bert"
    df4 = calculate_stats(prep_inf_times)
    df4["Flag"] = "Prep & Inf Time Total"
    df5 = calculate_stats(inference_times)
    df5["Flag"] = "Inference Time"

    dfs = pd.concat([df1,df2,df3,df5,df4])
    
    print(num_observations, ", ", dfs)
    return dfs

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'

print("_______________________________________________________________")
print("Inferencing Started...........")
if __name__=='__main__':
    ob_ct = 1  # Start with a single observation
    logging.info(STATS)
    temp_df = pd.DataFrame()
    while ob_ct <= 100000:
        temp = run_inference(ob_ct)
        temp["No_of_Observation"] = ob_ct
        temp_df = temp_df.append(temp)
        ob_ct *= 10
    print("Summary........")
    temp_df.to_csv("Results.csv")
    print(temp_df)
