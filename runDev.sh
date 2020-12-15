#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=3
batch_size=16
model_base_dir=all_outputs/saved_models

#xlm-roberta-large
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3  myDev.py\
  --model_name xlm-roberta-large --batch_size ${batch_size} \
  --model_save_dir=${model_base_dir}/xlm-roberta-large/kfold_feat_1 --add_features 1\
  --xlm_test_model_list  NP:xlm-roberta-large-b16-lr2e-05-dr0.3-wup0-0.8558  NP:xlm-roberta-large-b16-lr2e-05-dr0.3-wup0-0.8616\
    NP:xlm-roberta-large-b16-lr2e-05-dr0.3-wup0-0.8709 \
    NP:xlm-roberta-large-b16-lr2e-05-dr0.3-wup0-0.8614  NP:xlm-roberta-large-b16-lr2e-05-dr0.3-wup0-0.8663




