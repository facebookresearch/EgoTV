# measure accuracy and F1 scores over different test splits, run_ids and along different axes of complexity and ordering
import os
import sys

sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
sys.path.append(os.environ['CKPTS'])
from dataset_utils import *
from feature_extraction import *
from proScript.utils import GraphEditDistance
from end2end.rnn import RNNEncoder
from nesy_model import NeSyBase
from nesy_arguments import Arguments
from distributed_utils import *
import json
import math
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def test_model(test_loader):
    with torch.no_grad():
        for video_feats, graphs, labels, segment_labs, task_types, axis_stats in tqdm(iterate(test_loader), desc='Test'):
            preds, labels, axis_stats = model(video_feats, graphs, labels, task_types, axis_stats, train=False)
            labels = labels.type(torch.int)
            # test_metrics.update(preds=output, target=labels)
            assert len(axis_stats) == len(preds), "error in one or more t5 graph generations"
            for ind, axis_stat in enumerate(axis_stats):
                axis_metrics[axis_stat[0]][axis_stat[1]].update(preds=preds[ind].view(-1),
                                                                target=labels[ind].view(-1))
            test_metrics[split_type].update(preds=preds, target=labels)
        dist.barrier()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        # try:
        yield process_batch(data_batch, label_batch, frames_per_segment=args.fp_seg)
        # except TypeError:
        #     print('Skipping batch')
        #     continue


def process_batch(data_batch, label_batch, frames_per_segment):
    hypotheses = []
    video_features_batch = []  # transforms + visual model features
    labels = []
    task_types = []
    segment_labs_batch = []
    axis_stats_batch = []

    for filepath, label in zip(data_batch, label_batch):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        task_types.append(traj['task_type'])
        hypotheses.append(traj['template']['neg']) if label == '0' \
            else hypotheses.append(traj['template']['pos'])
        labels.append(float(label))

    # generate graphs for hypotheses
    with torch.no_grad():
        inputs_tokenized = tokenizer(hypotheses, return_tensors="pt",
                                     padding="longest",
                                     max_length=max_source_length,
                                     truncation=True)
        graphs_batch = t5_model.module.generate(inputs_tokenized["input_ids"].cuda(),
                                                attention_mask=inputs_tokenized["attention_mask"].cuda(),
                                                max_length=max_target_length,
                                                do_sample=False)  # greedy generation
        graphs_batch = tokenizer.batch_decode(graphs_batch, skip_special_tokens=True)
        graphs_batch = ([ged.pydot_to_nx(graph_str) for graph_str in graphs_batch], hypotheses)

    all_arguments = retrieve_query_args(graphs_batch[0])  # retrieve query arguments from the graph batch

    for sample_ind, (filepath, label) in enumerate(zip(data_batch, label_batch)):
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        complexity, ordering = task_axis_stats(task_type=traj['task_type'])
        axis_stats_batch.append((complexity, ordering))
        segment_labs, roi_bb, _ = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                         all_arguments[sample_ind],
                                                         positive=True if label == '1' else False)
        segment_labs_batch.append(segment_labs)
        video_frames, roi_frames = sample_vid_with_roi(filepath, args.sample_rate, roi_bb)
        video_frames, roi_frames = torch.stack(video_frames).cuda(), torch.stack(roi_frames).cuda()  # [t, c, h, w]
        # here b=1 since we are processing one video at a time
        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor='clip',
                                              feat_size=vid_feat_size,
                                              finetune=args.finetune).reshape(1, -1, vid_feat_size)
        roi_frames = extract_video_features(roi_frames.reshape(-1, 3, 224, 224), model=visual_model,
                                            feature_extractor='clip',
                                            feat_size=vid_feat_size,
                                            finetune=args.finetune).reshape(1, -1, 3 * vid_feat_size)
        b, t, _ = video_frames.shape
        num_segments = math.ceil(t / frames_per_segment)

        to_pad = num_segments * frames_per_segment - t
        # zero-padding to match the number of frames per segment
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)
        roi_frames = torch.cat((roi_frames, torch.zeros(b, to_pad, 3 * vid_feat_size).cuda()), dim=1)
        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)
        roi_frames = roi_frames.reshape(b * num_segments, frames_per_segment, 3 * vid_feat_size)

        # [num_segments, frames_per_segment, k*512]
        video_frames = torch.cat((video_frames, roi_frames), dim=-1)
        video_features_batch.append(video_frames)

    return video_features_batch, graphs_batch, torch.tensor(labels).cuda(), segment_labs_batch, task_types, axis_stats_batch


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
    ged = GraphEditDistance()
    args = Arguments()

    # text module to generate graph
    max_source_length = 80
    max_target_length = 300
    proscript_model_ckpt_path = os.path.join(os.environ['BASELINES'],
                                             'proScript/proscript_best_dsl_3.json')
    t5_model = T5ForConditionalGeneration.from_pretrained(proscript_model_ckpt_path)
    t5_model.cuda()
    t5_model = DDP(t5_model, device_ids=[local_rank])
    t5_model.eval()
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    visual_model, vid_feat_size = initiate_visual_module(feature_extractor='clip')
    visual_model.cuda()
    visual_model = nn.SyncBatchNorm.convert_sync_batchnorm(visual_model)
    visual_model = DDP(visual_model, device_ids=[local_rank])
    if not args.finetune:
        visual_model.eval()
    else:
        visual_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format('clip'))

    _, _, text_feat_size = initiate_text_module(feature_extractor='clip')
    text_model = visual_model  # for clip model

    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True),
                                F1Score(threshold=0.5, dist_sync_on_step=True)]).cuda()
    all_test_splits = ['sub_goal_composition', 'verb_noun_composition',
                       'context_verb_noun_composition', 'context_goal_composition', 'abstraction']

    # axis metrics measure accuracy and F1 along the axes of complexity and ordering (averaged over different run_ids)
    axis_metrics = {complexity: {ordering: metrics.clone(prefix='test_')
                                 for ordering in np.arange(0, 3, 1)} for complexity in np.arange(1, 4, 1)}

    axis_results = {complexity: {ordering: tuple()
                                 for ordering in np.arange(0, 3, 1)} for complexity in np.arange(1, 4, 1)}
    # test metrics measure accuracy, F1 for each test split (averaged over different run_ids)
    test_metrics = {k: metrics.clone(prefix='test_') for k in all_test_splits}

    test_logger = 'nesy_{}_log_test.txt'.format(args.fp_seg)
    logger_path = os.path.join(os.getcwd(), test_logger)
    test_log_file = open(logger_path, "w")
    axes_log_file = 'nesy_{}_axes.pkl'.format(args.fp_seg)
    axes_log_path = os.path.join(os.getcwd(), axes_log_file)
    model_val_acc, model_val_f1 = [], []

    for run_id in ['51']:
        ckpt_file = 'nesy_best_{}.pth'.format(str(args.run_id))
        model_ckpt_path = os.path.join(os.environ['CKPTS'], ckpt_file)
        train_log_file = 'nesy_log_{}.txt'.format(str(args.run_id))
        train_log_path = os.path.join(os.environ['CKPTS'], train_log_file)
        best_val_acc, best_val_f1 = train_log_process(train_log_path)
        model_val_acc.append(best_val_acc)
        model_val_f1.append(best_val_f1)

        hsize = 150
        model = NeSyBase(vid_embed_size=vid_feat_size, hsize=hsize, rnn_enc=RNNEncoder, text_model=text_model)
        model.load_state_dict(torch.load(model_ckpt_path))
        model.cuda()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
        model.eval()

        for split_type in all_test_splits:
            if is_main_process():
                print('======== run_id: {} | split type: {} =============='.format(int(run_id), split_type))
            path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', split_type)
            if args.preprocess:
                preprocess_dataset(path, args.split_type)
            test_set = CustomDataset(data_path=path)
            test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                                     num_workers=args.num_workers, shuffle=False)
            test_model(test_loader)

    # dist.barrier()
    # if is_main_process():
    for com_key, com_val in axis_metrics.items():
        for ord_key, ord_val in com_val.items():
            try:
                acc, f1 = ord_val['Accuracy'].compute(), ord_val['F1Score'].compute()
                axis_results[com_key][ord_key] = tuple((acc.item(), f1.item()))
            except:
                axis_results[com_key][ord_key] = tuple((0, 0))
    print('Axis Results: {}'.format(axis_results))
    pkl.dump(axis_results, open(axes_log_path, 'wb'))

    mean_acc, mean_f1 = str(np.array(model_val_acc).mean()), str(np.array(model_val_f1).mean())
    print('Split Type: validation | Acc: {} | F1: {}'.format(mean_acc, mean_f1))
    test_log_file.write('Split: ' + 'validation' + ' | Test Acc: ' + mean_acc + ' | Test F1: ' + mean_f1 + "\n")
    test_log_file.flush()
    for split_type in all_test_splits:
        test_acc, test_f1 = test_metrics[split_type]['Accuracy'].compute(), \
                            test_metrics[split_type]['F1Score'].compute()
        print('Split Type: {} | Acc: {} | F1: {}'.format(split_type, str(test_acc.item()), str(test_f1.item())))
        test_log_file.write('Split: ' + split_type + ' | Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
        test_log_file.flush()
    test_log_file.close()
    print('Done!')
