#!/bin/bash
# clip4clip baseline
source activate alfred_env
export DATA_ROOT=/fb-agios-acai-efs/dataset
export BASELINES=$(pwd)/VisionLangaugeGrounding/baselines
cd $BASELINES/end2end

echo "=================== clip4clip baseline ==================="
for run_id in 1 2 3
  do
  echo "clip4clip run_id:" $run_id
    for var in 'seqLSTM' 'tightTransfer'
      do
        echo "clip4clip sim type:" $var
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 clip4clip.py --num_workers 2 --split_type 'train' --batch_size 64 --sample_rate 3 --run_id $run_id --epochs 40 --lr 1e-3 --sim_type $var
        #	ps -ef | grep 'clip4clip.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9
      done
  done


