# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse

def Arguments():
    parser = argparse.ArgumentParser(description="baseline")
    parser.add_argument('--sample_rate', type=int, default=5,
                        help='video sub-sample rate (higher sample rate -> fewer frames)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs of training')
    parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloaders')
    parser.add_argument('--data_split', type=float, default=0.8, help='train-val split')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training, validation, test; '
                                                                  'batch size is divided across the number of workers')
    # '''<command> --preprocess''' to set preprocess
    parser.add_argument('--preprocess', action='store_true',
                        help='process dataset before training, validation, testing')
    parser.add_argument('--split_type', type=str, default='train',
                        help='dataset split on which model will run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--visual_feature_extractor', type=str, default='resnet',
                        choices=['I3D', 'resnet', 'mvit', 'S3D', 'clip'],
                        help='I3D or resnet features for video (premise)')
    parser.add_argument('--text_feature_extractor', type=str, default='bert', choices=['bert', 'glove', 'clip'],
                        help='bert or glove features for text (hypothesis)')
    parser.add_argument('--pretrained_mvit', type=str, default=True,
                        help='if True, load pretrained weights for MViT from Kinetics400 mvit model')
    # '''<command> --finetune''' to set finetune
    parser.add_argument('--finetune', action='store_true', help='whether to finetune resnet/I3D model '
                                                                     'in the specific setup')
    # '''<command> --attention''' to set attention
    parser.add_argument('--attention', action='store_true', help='to use bidaf attention ?')
    parser.add_argument('--resume', action='store_true', help='to resume training from a previously save checkpoint')
    parser.add_argument('--run_id', type=int, default=5, required=False, help='run_id of the model run')

    # baselines
    parser.add_argument('--sim_type', type=str, choices=['meanPool', 'seqLSTM', 'tightTransfer', 'hitchHiker'],
                        help='similarity type for CLIP4Clip/Clip HitchHiker baseline')
    return parser.parse_args()
