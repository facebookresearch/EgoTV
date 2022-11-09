import argparse


def Arguments():
    parser = argparse.ArgumentParser(description="proScript")
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--num_workers', type=int, default=0, help='workers for dataloaders')
    parser.add_argument('--data_split', type=float, default=0.9, help='train-val split')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, validation, test; '
                                                                   'batch size is divided across the number of workers')
    parser.add_argument('--preprocess', action='store_true', required=True,
                        help='process dataset before training, validation, testing')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--run_id', type=int, default=5, required=True, help='run_id of the model run')
    parser.add_argument('--test_split', type=str, default='sub_goal_composition', required=True,
                        help='test split graph generation')
    return parser.parse_args()
