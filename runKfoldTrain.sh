#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0
batch_size=16
lr=2e-5
early_stop=5
warm_up_steps=0
model_base_dir='all_outputs/saved_models'

#xlm-roberta-large
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3  myKfoldTrain.py  --model_name roberta-large\
        --batch_size ${batch_size} --lr ${lr} --early_stop ${early_stop} --warm_up_steps ${warm_up_steps} \
        --model_save_dir=${model_base_dir}/roberta-large/kfold_feat_1  --add_features 1