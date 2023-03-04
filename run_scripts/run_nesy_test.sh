#!/bin/bash
# nsg
source activate alfred_env
export DATA_ROOT=/fb-agios-acai-efs/dataset
export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
cd VisionLangaugeGrounding/nsg

vis_feat='coca'
text_feat='coca'
context_encoder='mha'
run_id=1
echo "=================== NSG Model Test ==================="
for split in 'sub_goal_composition' 'verb_noun_composition' 'context_verb_noun_composition' 'context_goal_composition' 'abstraction'
  do
  echo "===================================================="
  echo $run_id " | " $split " | " $vis_feat " | " $text_feat " | " $context_encoder
  echo "===================================================="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 nesy_test.py --num_workers 2 --split_type $split --batch_size 64 --sample_rate 2 --run_id 50 --fp_seg 20 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --context_encoder $context_encoder
done