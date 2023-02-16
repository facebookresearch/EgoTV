import argparse


def Arguments():
    parser = argparse.ArgumentParser(description="nesy-baseline")
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='video sub-sample rate (higher sample rate -> fewer frames)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs of training')
    parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloaders')
    parser.add_argument('--data_split', type=float, default=0.8, help='train-val split')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training, validation, test; '
                                                                  'batch size is divided across the number of workers')
    # '''<command> --preprocess''' to set preprocess
    parser.add_argument('--preprocess', action='store_true',
                        help='process dataset before training, validation, testing')
    parser.add_argument('--split_type', type=str, default='context_goal_composition',
                        help='dataset split on which model will run')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pretrained_mvit', type=str, default=True,
                        help='if True, load pretrained weights for MViT from Kinetics400 mvit model')
    parser.add_argument('--text_feature_extractor', type=str, default='clip', choices=['clip'],
                        help='bert or glove features graph arguments')
    parser.add_argument('--fp_seg', type=int, default=20, help='frames per segment')
    # '''<command> --finetune''' to set finetune
    parser.add_argument('--finetune', action='store_true', help='whether to finetune clip model '
                                                                'in the specific setup')
    parser.add_argument('--resume', action='store_true', help='to resume training from a previously save checkpoint')
    parser.add_argument('--run_id', type=int, default=50, required=False, help='run_id of the model run')
    return parser.parse_args()
