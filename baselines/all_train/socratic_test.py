import os
import sys
# os.environ['DATA_ROOT'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
# os.environ['BASELINES'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])

from dataset_utils import *
from arguments import Arguments
import numpy as np
from tqdm import tqdm
from torchmetrics import MetricCollection, Accuracy, F1Score


def test_model(test_loader):
    for data_batch, label_batch in tqdm(test_loader, desc='Test'):
        stage_1(data_batch, label_batch)


def stage_1(data_batch, label_batch):
    """
    extract raw video captions
    A. decompose the video into clips
    B. sub-sample k-frames from each clips
    C. detect vocab objects from each frame
    :return:
    """
    detected_objs = {}
    for filepath, label in zip(data_batch, label_batch):
        # ============ sampling frames from the video ============ #
        # type='coca' returns PIL images without processing
        video_frames = sample_vid(filepath, args.sample_rate, type='coca')  # list of PIL images        print('hi')
        clips = [video_frames[x:x+15] for x in range(0, len(video_frames), 15)]
        # sample k-frames from each clip
        detected_objs[filepath] = []
        for clip_ind, clip in enumerate(clips):
            k = 3 if len(clip) > 3 else len(clip)
            sampled_indices = np.sort(np.random.choice(len(clip), k, replace=False))
            for ind in sampled_indices:
                sampled_frame = clip[ind]
                '''
                detect objects in each frame and store them in a list 
                clip_objs.append(model_run() else None)
                '''
            detected_objs[filepath].append((clip_ind, ','.join(clip_objs)))
    with open(stage_1_outfile, "w") as outfile:
        outfile.write(detected_objs)


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
    # dist_url = "env://"  # default
    # # only works with torch.distributed.launch // torch.run
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    # local_rank = int(os.environ['LOCAL_RANK'])
    # dist.init_process_group(
    #     backend="nccl",
    #     init_method=dist_url,
    #     world_size=world_size,
    #     rank=rank)
    # # this will make all .cuda() calls work properly
    # torch.cuda.set_device(local_rank)
    # # synchronizes all the threads to reach this point before moving on
    # dist.barrier()
    args = Arguments()
    path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.split_type)
    '''import RegionCLIP model and checkpoint (make it distributed if possible)'''

    # file for storing clip captions
    stage_1_outfile = os.path.join(os.getcwd(), '{}_stage_1.json'.format(args.split_type))

    if args.preprocess:
        preprocess_dataset(path, args.split_type)
    test_set = CustomDataset(data_path=path)
    # test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)


    metrics = MetricCollection([Accuracy(dist_sync_on_step=True, task='binary'),
                                F1Score(dist_sync_on_step=True, task='binary')])
    test_metrics = metrics.clone(prefix='test_')

    best_acc = 0.
    # test_loader.sampler.set_epoch(0)
    test_model(test_loader)
    # test_metrics.reset()
    # cleanup()
    print('Done!')
