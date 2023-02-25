# pip install -U sentence-transformers
import os
import sys
# os.environ['DATA_ROOT'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
# os.environ['BASELINES'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from distributed_utils import *
from arguments import Arguments
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection, Accuracy, F1Score


def test_model(test_loader):
    with torch.no_grad():
        for text_bath, video_caption_batch, label_batch in tqdm(iterate(test_loader), desc='Test'):
            # text_enc, video_caption_enc = stage_1(data_batch, label_batch)
            sim_scores = torch.cosine_similarity(text_bath, video_caption_batch)
            labels = label_batch.type(torch.int)
            test_metrics.update(preds=sim_scores, target=labels)
            # print(list(zip(sim_scores, label_batch)))
        dist.barrier()
        test_acc, test_f1 = list(test_metrics.compute().values())
        dist.barrier()
        if is_main_process():
            print('Test Acc: {} | Test F1: {}'.format(test_acc, test_f1))
            log_file.write('Test Acc: ' + str(test_acc.item()) + ' | Test F1: ' + str(test_f1.item()) + "\n")
            log_file.flush()


def iterate(dataloader):
    for data_batch, label_batch in tqdm(dataloader):
        yield stage_1(data_batch, label_batch)


def action_map_to_simple(action):
    action_map = {'HeatObject': 'heat', 'SliceObject': 'slice',
                  'CoolObject': 'cool', 'CleanObject': 'clean',
                  'PutObject': 'put', 'PickupObject': 'pick',
                  'GotoLocation': 'go to', 'NoOp': ''}
    return action_map[action]


def stage_1(data_batch, label_batch, frames_per_segment=20):
    """
    extract raw video captions
    A. decompose the video into clips
    B. sub-sample k-frames from each clips
    C. detect vocab objects from each frame
    :return:
    """
    hypotheses = []
    video_captions = []
    labels = []
    for filepath, label in zip(data_batch, label_batch):
        labels.append(float(label))
        # ============ sampling frames from the video ============ #
        # type='coca' returns PIL images without processing
        # video_frames = sample_vid(filepath, args.sample_rate, type='coca')  # list of PIL images
        traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
        hypotheses.append(traj['template']['neg']) if label == '0' \
            else hypotheses.append(traj['template']['pos'])

        # list of segment actions and detected objects for each clip in the video (len of list = # clips)
        segment_actions, detected_objs = extract_segment_labels(traj, args.sample_rate, frames_per_segment,
                                                         action_args=[], socratic=True)
        # clips = [video_frames[x:x+frames_per_segment] for x in range(0, len(video_frames), frames_per_segment)]
        video_caption = ''
        for clip_ind, (action, object) in enumerate(zip(segment_actions, detected_objs)):
            object_str = ', '.join(object) if type(object) == list else object
            video_caption += 'Section: {}. Place: kitchen. Objects: {}. Activity: {} {}\n'.format(clip_ind+1,
                                                                                               object_str,
                                                                                               action_map_to_simple(
                                                                                                   action),
                                                                                               object_str.split(' ')[0])
        # print(video_caption)
        video_captions.append(video_caption)
    return torch.tensor(roberta_model.module.encode(hypotheses)).cuda(), \
           torch.tensor(roberta_model.module.encode(video_captions)).cuda(), \
           torch.tensor(labels).cuda()


def stage_2():
    """
    process_captions_for_clips
    :return:
    """
    pass


def stage_3():
    """
    get final video captions
    :return:
    """
    pass


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
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    logger_filename = 'socratic_{}_{}.txt'.format(args.split_type, str(args.run_id))
    logger_path = os.path.join(os.environ['BASELINES'], logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    # Distilroberta for sentence embedding (https://arxiv.org/abs/1907.11692)
    roberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    roberta_model.cuda()
    roberta_model = DDP(roberta_model, device_ids=[local_rank])
    roberta_model.eval()

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, shuffle=False)


    metrics = MetricCollection([Accuracy(threshold=0.24, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.24, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = metrics.clone(prefix='test_')

    best_acc = 0.
    test_loader.sampler.set_epoch(0)
    test_model(test_loader)
    test_metrics.reset()
    cleanup()
    print('Done!')
