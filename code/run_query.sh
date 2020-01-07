#!/usr/bin/env bash

data_file_train="data/model_data_train.query"
data_file_test="data/model_data_test.query"

vocab_file="data/vocab.query"
seg_label_file="data/label_seg_grain_idx.query"
coarse_label_file="data/label_coarse_grain_idx.query"
fine_label_file="data/label_fine_grain_idx.query"
encoding="utf8"

fine_num_classes=57
coarse_num_classes=7
seg_num_classes=3

max_seq_len=21
vocab_size=1265
embedding_dim=100

num_epochs=10
batch_size=32
dropout_keep_prob=1.0
eval_step=1

max_grad_norm=5.0
learning_rate=0.001
l2_reg_lambda=0.0

lstm_dim=100
layer_size=3

is_multi_task="True" # or "False"

python train.py \
    --data_file_train=${data_file_train} \
    --data_file_test=${data_file_test} \
    --vocab_file=${vocab_file} \
    --seg_label_file=${seg_label_file} \
    --coarse_label_file=${coarse_label_file} \
    --fine_label_file=${fine_label_file} \
    --encoding=${encoding} \
    --max_seq_len=${max_seq_len} \
    --seg_num_classes=${seg_num_classes} \
    --coarse_num_classes=${coarse_num_classes} \
    --fine_num_classes=${fine_num_classes} \
    --vocab_size=${vocab_size} \
    --embedding_dim=${embedding_dim} \
    --num_epochs=${num_epochs} \
    --batch_size=${batch_size} \
    --dropout_keep_prob=${dropout_keep_prob} \
    --eval_step=${eval_step} \
    --max_grad_norm=${max_grad_norm} \
    --learning_rate=${learning_rate} \
    --l2_reg_lambda=${l2_reg_lambda} \
    --lstm_dim=${lstm_dim} \
    --layer_size=${layer_size} \
    --is_multi_task=${is_multi_task}