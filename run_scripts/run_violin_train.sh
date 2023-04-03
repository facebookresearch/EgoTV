#!/bin/bash

source activate alfred_env
export DATA_ROOT=/fb-agios-acai-efs/dataset
export BASELINES=../EgoTV/baselines
cd $BASELINES/all_train

run_id=3
vis_feat='I3D'
echo "=================== i3d, bert/glove, no attention ==================="
for var in 'bert' 'glove'
do
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m torch.distributed.launch --nproc_per_node=7 violin.py --num_workers 4 --split_type 'train' --batch_size 64 --sample_rate 3 --visual_feature_extractor $vis_feat --text_feature_extractor $var --run_id $run_id --epochs 40 --lr 1e-3
	ps -ef | grep 'violin.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
done


