import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from dataset_utils import *
from feature_extraction import *
from violin_base import ViolinBase
from i3d.pytorch_i3d import InceptionI3d
from mvit_tx.mvit import mvit_v2_s
from arguments import Arguments
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18 as resnet
from transformers import DistilBertModel, DistilBertTokenizer
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe


def train_epoch(model, train_loader, val_loader, epoch, previous_best_acc):
    model.train()
    if args.finetune:
        visual_model.train()
    train_loss = []
    for video_features, hypotheses, labels in tqdm(iterate(train_loader), desc='Train'):
        output = model(video_features, hypotheses)
        loss = bce_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        # model.state_dict(keep_vars=True)
        # {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
        optimizer.step()
        # print('Loss: {}'.format(loss.item()))
        train_loss.append(loss.item())
        labels = labels.type(torch.int).cuda()
        train_metrics.update(preds=output, target=labels)

    acc, f1 = train_metrics['Accuracy'].compute(), train_metrics['F1Score'].compute()
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
    dist.barrier()
    val_acc, val_f1 = validate(model, val_loader=val_loader)
    dist.barrier()
    if is_main_process():
        print('Epoch: {} | Train Acc: {} | Val Acc: {} | Train F1: {} | Val F1: {}'.format(
            epoch, acc, val_acc, f1, val_f1))
        log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +
                       ' | Val Acc: ' + str(val_acc.item()) + ' | Train F1: ' + str(f1.item()) +
                       ' | Val F1: ' + str(val_f1.item()) + "\n")
        if val_acc > torch.tensor(previous_best_acc).cuda():
            previous_best_acc = val_acc.item()
            print('============== Saving best model(s) ================')
            torch.save(model.module.state_dict(), model_ckpt_path)
            if args.finetune:
                torch.save(visual_model.module.state_dict(), visual_model_ckpt_path)
        log_file.flush()
    return previous_best_acc


def validate(model, val_loader):
    model.eval()
    visual_model.eval()
    with torch.no_grad():
        for video_features, hypotheses, labels in tqdm(iterate(val_loader), desc='Validation'):
            output = model(video_features, hypotheses)
            labels = labels.type(torch.int).cuda()
            val_metrics.update(preds=output, target=labels)
    return val_metrics['Accuracy'].compute(), val_metrics['F1Score'].compute()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield process_batch(data_batch, label_batch)


def process_batch(data_batch, label_batch):
    hypotheses, labels = [], []

    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        '''loading trajectory data'''
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))

        '''sampling frames from the video'''
        # video_frames = [cv2.imread(frame) for frame in glob.glob(os.path.join(filepath, 'raw_images') + "/*.png")]
        video_frames = sample_vid(filepath, args.sample_rate)
        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]
        '''process video features using resnet/i3d/mvit'''
        # video_features_batch is a tuple of (video_features, video_lengths)
        video_features_batch = extract_video_features(video_frames,
                                                      model=visual_model,
                                                      feature_extractor=args.visual_feature_extractor,
                                                      feat_size=vid_feat_size,
                                                      finetune=args.finetune)

        '''process natural language hypothesis using bert/glove'''
        if label == '0':
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['neg'].split(' ')])
        else:
            clean_string = ' '.join([clean_str(word).lower() for word in traj['template']['pos'].split(' ')])
        hypotheses.append(clean_string)
        hypotheses = extract_text_features(hypotheses,
                                           model=text_model,
                                           feature_extractor=args.text_feature_extractor,
                                           tokenizer=tokenizer,
                                           glove_vec=global_vectors if args.text_feature_extractor=='glove' else None)
    return video_features_batch, hypotheses, torch.tensor(labels).cuda()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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
    if args.split_type == 'train':
        path = os.path.join(os.environ['DATA_ROOT'], args.split_type)
    else:
        path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    if args.attention:
        ckpt_file = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_attention_best_' + str(args.run_id) + '.pth'
    else:
        ckpt_file = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_best_' + str(args.run_id) + '.pth'
    model_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    hsize1 = 150
    hsize2 = 300

    if args.attention:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_attention_log_' + str(args.run_id) + '.txt'
    else:
        logger_filename = 'violin_' + args.visual_feature_extractor + '_' + \
                          args.text_feature_extractor + '_log_' + str(args.run_id) + '.txt'

    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        #TODO: automate [if not present json_file, preprocess must be done]
        preprocess_dataset(path, args.split_type)

    dataset = CustomDataset(data_path=path)
    train_size = int(args.data_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_sampler, val_sampler = DistributedSampler(dataset=train_set, shuffle=True), \
                                 DistributedSampler(dataset=val_set, shuffle=True)
    train_loader, val_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                          num_workers=args.num_workers, pin_memory=True), \
                               DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                          num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # distil bert model
    # TODO: also try TinyBert
    if args.text_feature_extractor == 'bert':
        embed_size = 768
        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
        # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
        text_model = DDP(text_model, device_ids=[local_rank])
        # TODO: get vocabulary file for bert tokenizer: BertTokenizer(vocab_file=?)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text_model.eval()
    elif args.text_feature_extractor == 'glove':
        embed_size = 300
        tokenizer = get_tokenizer("basic_english")
        global_vectors = GloVe(name='840B', dim=embed_size)
    else:
        raise NotImplementedError

    # visual feature extractor for the video
    if args.visual_feature_extractor == 'resnet':
        # resnet model
        vid_feat_size = 512  # 512 for resnet 18, 34; 2048 for resnet50, 101
        visual_model = resnet(pretrained=True)
        visual_model = nn.Sequential(*list(visual_model.children())[:-1])
    elif args.visual_feature_extractor == 'i3d':
        # i3d model
        vid_feat_size = 1024
        kinetics_pretrained = 'i3d/rgb_imagenet.pt'
        visual_model = InceptionI3d(400, in_channels=3)
        visual_model.load_state_dict(torch.load(os.path.join(os.environ['BASELINES'], kinetics_pretrained)))
        visual_model.replace_logits(157)
    elif args.visual_feature_extractor == 'mvit':
        # MViT-S ("https://arxiv.org/pdf/2104.11227.pdf")
        vid_feat_size = 768
        visual_model = mvit_v2_s()
    else:
        raise NotImplementedError
    visual_model.cuda()
    visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])

    if not args.finetune:
        visual_model.eval()
    else:
        visual_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format(args.visual_feature_extractor))

    # violin base model
    model = ViolinBase(hsize1=hsize1, hsize2=hsize2, embed_size=embed_size, vid_feat_size=vid_feat_size,
                       attention=args.attention).cuda()
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])

    all_params = list(model.parameters())
    if args.finetune:
        all_params += list(visual_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCELoss()
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)]).cuda()
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        # enable different shuffling with set_epoch (uses it to set seed = epoch)
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        best_acc = train_epoch(model, train_loader, val_loader, epoch=epoch, previous_best_acc=best_acc)
        train_metrics.reset()
        val_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
