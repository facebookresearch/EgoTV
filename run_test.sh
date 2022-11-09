#!/bin/bash

source activate alfred_env
export DATA_ROOT=/fb-agios-acai-efs/dataset
export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
cd $BASELINES/end2end/violin
run_id=1
vis_feat='i3d'
echo "=================== i3d, bert, no attention ==================="
text_feat='bert'
for var in 'sub_goal_composition' 'verb_noun_composition' 'context_verb_noun_composition' 'context_goal_composition' 'abstraction'
# for var in 'context_goal_composition'
do
  echo "===================================================="
	echo $var
	echo "===================================================="
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m torch.distributed.launch --nproc_per_node=7 test_baseline.py --num_workers 4 --split_type $var --batch_size 64 --sample_rate 3 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --run_id $run_id --attention
	ps -ef | grep 'test_baseline.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
done

echo "=================== i3d, glove, no attention ==================="
text_feat='glove'
for var in 'sub_goal_composition' 'verb_noun_composition' 'context_verb_noun_composition' 'context_goal_composition' 'abstraction'
# for var in 'context_goal_composition'
do
  echo "===================================================="
	echo $var
	echo "===================================================="
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 python -m torch.distributed.launch --nproc_per_node=7 test_baseline.py --num_workers 4 --split_type $var --batch_size 64 --sample_rate 3 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --run_id $run_id --attention
	ps -ef | grep 'test_baseline.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
done


