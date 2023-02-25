import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from feature_extraction import *
from VIOLIN.violin_model import ModelBase
from distributed_utils import *
from arguments import Arguments
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler


def test_model(test_loader):
    with torch.no_grad():
        for video_feats, text_feats, labels in tqdm(iterate(test_loader), desc='Test'):
            output = model(video_feats, text_feats)
            labels = labels.type(torch.int).cuda()
            test_metrics.update(preds=output, target=labels)
        dist.barrier()
        test_acc, test_f1 = list(test_metrics.compute().values())
        dist.barrier()
        if is_main_process():
            print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
            log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
            log_file.flush()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses, labels = [], []
    video_feat_batch, vid_lengths = [], []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        '''loading trajectory data'''
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        '''sampling frames from the video'''
        # video_frames = [cv2.imread(frame) for frame in glob.glob(os.path.join(filepath, 'raw_images') + "/*.png")]
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]

        '''process video features using resnet/I3D/mvit'''
        video_feats = extract_video_features(video_frames,
                                             model=visual_model,
                                             feature_extractor=args.visual_feature_extractor,
                                             feat_size=vid_feat_size,
                                             finetune=args.finetune,
                                             test=True)  # [max_seq_len, batch_size, embed_dim]
        video_feat_batch.append(video_feats)
        vid_lengths.append(len(video_feats))

        '''process natural language hypothesis using bert/glove'''
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)

    # tuple: (features, vid_lens) |  sizes: ([batch_size, max_seq_len, embed_dim], [batch_size])
    video_feat_batch = (pad_sequence(video_feat_batch).permute(1, 0, 2).contiguous(),
                        torch.tensor(vid_lengths))
    # tuple: (features, text_len) |  sizes: ([batch_size, max_seq_len, embed_dim], [batch_size])
    text_feat_batch = extract_text_features(hypotheses,
                                            model=text_model,
                                            feature_extractor=args.text_feature_extractor,
                                            tokenizer=tokenizer)
    return video_feat_batch, text_feat_batch, torch.tensor(labels).cuda()


if __name__ == '__main__':
    dist_url = "env://"  # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

    args = Arguments()
    print("============ attention is {} ============".format(str(args.attention)))
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    ckpt_file = 'violin_{}_{}{}_best_{}.pth'.format(args.visual_feature_extractor,
                                                     args.text_feature_extractor,
                                                     '_attention' if args.attention else '',
                                                     str(args.run_id))
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    logger_filename = 'violin_{}_{}{}_log_test_{}.txt'.format(args.visual_feature_extractor,
                                                          args.text_feature_extractor,
                                                          '_attention' if args.attention else '',
                                                          str(args.run_id))
    logger_path = os.path.join(os.environ['BASELINES'], logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, shuffle=False)

    # text feature extractor for the hypothesis
    text_model, tokenizer, embed_size = initiate_text_module(args.text_feature_extractor)
    if args.text_feature_extractor == 'bert':
        text_model.cuda()
        text_model = DDP(text_model, device_ids=[local_rank])
        text_model.eval()

    # visual feature extractor for the video
    visual_model, vid_feat_size = initiate_visual_module(args.visual_feature_extractor,
                                                         args.pretrained_mvit)
    visual_model.cuda()
    if args.visual_feature_extractor not in ['clip', 'mvit']:
        visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    visual_model.eval()

    # violin base model
    hsize1 = 150
    hsize2 = 300
    model = ModelBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size, vid_feat_size=vid_feat_size,
                       attention=args.attention)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    test_loader.sampler.set_epoch(0)
    test_model(test_loader)
    test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
