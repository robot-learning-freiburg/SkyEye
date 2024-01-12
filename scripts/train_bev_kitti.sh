#!/bin/bash

# Path to the Python virtual environment
python_venv={PATH OF THE PYTHON VENV}
source "$python_venv/bin/activate";
alias python="$python_venv/bin/python3";

# Uncomment if using WANDB
# export WANDB_API_KEY=""
# wandb login --relogin $WANDB_API_KEY;
# ulimit -n 100000;

CUDA_VISIBLE_DEVICES="{CUDA_GPU_IDS}" \
OMP_NUM_THREADS=4 \
torchrun --nproc_per_node="{NUM GPUS}" --master_addr='{IP ADDR}' --master_port={PORT NUM} train_bev.py \
                            --run_name={NAME OF THE RUN} \
                            --project_root_dir={PATH OF THE skyeye FOLDER} \
                            --seam_root_dir={PATH OF THE KITTI360 SEAMLESS DIRECTORY} \
                            --dataset_root_dir={PATH OF THE KITTI360 DATASET} \
                            --mode=train \
                            --use_wandb=False \
                            --defaults_config=kitti_defaults.ini \
                            --config=kitti_bev_1.ini \
                            --bev_percentage=1 \
                            --pre_train \
                            body:{PATH OF LASTEST SAVED MODEL FROM THE PRETRAINING STEP} \
                            voxel_grid:{PATH OF LASTEST SAVED MODEL FROM THE PRETRAINING STEP} \
                            --comment="SkyEye BEV Training. Using 1% of BEV Pseudolabels";

deactivate;