#!/bin/bash

SEED=7656

# 9661:

python swag.py \
--data="MRI" \
--model_path="../notebooks/saved_models/MRI-745_20210127" \
--checkpoint_path="../notebooks/saved_models/MRI_871_20210217/cp.ckpt" \
--save_path="../notebooks/saved_models/SWAG_A_S-${SEED}/" \
--epochs=600 \
--wd=1e-5 \
--lr_init=1e-4 \
--momentum=0.9 \
--swag_start=1 \
--swag_interval=1 \
--swag_lr=1e-4 \
--subspace='pca' \
--seed=${SEED}