#!/bin/bash

python train_mode_connectivity.py \
--data="MRI" \
--model="BaseNet" \
--curve='PolyChain' \
--num_bends=3 \
--init_start="../notebooks/saved_models/SWAG_A_S-6567" \
--init_end="../notebooks/saved_models/SWAG_B_S-7656" \
--fix_start \
--fix_end \
--lr=5e-5 \
--wd=1e-5 \
--epochs=1000 \
--seed=6567