#!/bin/bash
# videoclip baseline
source activate alfred_env
export DATA_ROOT=$(pwd)/EgoTV/dataset
export BASELINES=$(pwd)/EgoTV/baselines
export S3D=$BASELINES/S3D
cd $BASELINES/all_train

echo "=================== VideoCLIP baseline ==================="
for run_id in 1 2 3
  do
    echo "VideoCLIP run_id:" $run_id
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 videoclip.py --num_workers 3 --split_type 'train' --batch_size 32 --sample_rate 3 --run_id $run_id --epochs 40 --lr 1e-3
  done


