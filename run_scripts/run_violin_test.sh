#!/bin/bash

source activate alfred_env
export DATA_ROOT=$(pwd)/EgoTV/dataset
export BASELINES=$(pwd)/EgoTV/baselines
cd $BASELINES/all_train

run_id=1
vis_feat='I3D'
echo "=================== i3d, bert, no attention ==================="
text_feat='bert'
for var in 'novel_tasks' 'novel_steps' 'novel_scenes' 'abstraction'
do
  echo "===================================================="
	echo $var
	echo "===================================================="
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m torch.distributed.launch --nproc_per_node=7 violin_test.py --num_workers 4 --split_type $var --batch_size 64 --sample_rate 3 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --run_id $run_id --attention
	ps -ef | grep 'violin_test.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
done

echo "=================== i3d, glove, no attention ==================="
text_feat='glove'
for var in 'novel_tasks' 'novel_steps' 'novel_scenes' 'abstraction'
do
  echo "===================================================="
	echo $var
	echo "===================================================="
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m torch.distributed.launch --nproc_per_node=7 violin_test.py --num_workers 4 --split_type $var --batch_size 64 --sample_rate 3 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --run_id $run_id --attention
	ps -ef | grep 'violin_test.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
done


