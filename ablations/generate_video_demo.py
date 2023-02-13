import os
import sys

os.environ['DATA_ROOT'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/alfred/gen/dataset'
os.environ['BASELINES'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/baselines'
os.environ['NESY'] = '/home/rishihazra/PycharmProjects/VisionLangaugeGrounding/nesy'
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
sys.path.append(os.environ['NESY'])
from nesy_arguments import Arguments
from proScript.utils import GraphEditDistance
from dataset_utils import *
from distributed_utils import *
from feature_extraction import *
import json
import numpy as np
import math
import cv2
from itertools import product


def process(filepath, frames_per_segment):
    # traj = json.load(open(os.path.join(filepath, 'traj_data.json'), 'r'))
    # video_frames = sample_vid(filepath, args.sample_rate)
    # for i, frame in enumerate(video_frames):
    #     cv2.imwrite(os.path.join(os.getcwd(), 'imgs_save', '{}.png'.format(i)), frame)
    t = 257
    num_segments = math.ceil(t / frames_per_segment)
    # to_pad = num_segments * frames_per_segment - t
    labels = [1]
    action_pred_labels = [0, 3, 2]
    action_true_labels = [0, 3, 4]
    segment_labs = [['GotoLocation', 'GotoLocation', 'GotoLocation', 'GotoLocation', 'PickupObject',
                     'GotoLocation', 'HeatObject', 'HeatObject', 'GotoLocation', 'CleanObject', 'PutObject',
                     'SliceObject', 'PickupObject']]

    pred_alignment = [[('Step 1 StateQuery(apple,heat)', 7),
                           ('Step 2 StateQuery(apple,clean)', 9),
                           ('Step 3 RelationQuery(apple,knife,slice)', 10)]]
    # img_files = ['{}_{}.png'.format(i, j) for i, j in product(range(0, 4), range(0, 21, 5)) if not (i == 0 and j != 6)]

    pred_ind = [0, 6]
    prev_pred_ind = 0
    change_flag = False
    graph_img_files = []
    i = 0
    for _ in range(t):
        while i // frames_per_segment == pred_alignment[0][prev_pred_ind][1]:
            pred_ind[0] = prev_pred_ind + 1
            if i % frames_per_segment == 0:
                pred_ind[1] = 0
            else:
                if i // 5 > 0 and i % 5 == 0:
                    pred_ind[1] += 5
            change_flag = True
            graph_img_files.append('_'.join([str(k) for k in pred_ind]) + '.png')
            i += 1
        if change_flag:
            if pred_ind[0] < len(pred_alignment[0]):
                prev_pred_ind += 1
            change_flag = False
        if pred_ind[0] > 0:
            pred_ind[1] = 20
        else:
            pred_ind[1] = 6
        if i != 200:
            graph_img_files.append('_'.join([str(k) for k in pred_ind]) + '.png')
            i += 1
        else:
            pass

    for ind_save, graph_img_file in enumerate(graph_img_files):
        graph_img = cv2.imread(os.path.join(os.getcwd(), graph_img_file))
        out_file = os.path.join(os.getcwd(), 'graph_imgs_save', str(ind_save) + '.png')
        cv2.imwrite(out_file, graph_img)

    combined_img_files = ['0_0.png', '0_1.png', '0_2.png', '0_3.png', '0_4.png', '0_5.png', '0_6.png'] +\
                      [str(i)+'.png' for i in np.arange(1, 251)]
    for ind, file_name in enumerate(combined_img_files):
        graph_img_file = os.path.join(os.getcwd(), 'graph_imgs_save', file_name)
        vid_img_file = os.path.join(os.getcwd(), 'imgs_save', file_name)
        graph_img = cv2.imread(graph_img_file)
        vid_img = cv2.imread(vid_img_file)
        cv2.imwrite(os.path.join(os.getcwd(), 'combined_imgs_save', '{}.png'.format(ind)),
                    np.concatenate((vid_img, graph_img), axis=1))
    #
    # video = cv2.VideoWriter(os.path.join(os.getcwd(), 'out1.avi'), 0, 5, (850, 600))
    #
    # for ind in np.arange(0, 257):
    #     im = cv2.imread(os.path.join(os.getcwd(), 'combined_imgs_save', '{}.png'.format(ind)))
    #     video.write(im)
    #
    # cv2.destroyAllWindows()
    # video.release()
#         video_frames = sample_vid(filepath, args.sample_rate)


if __name__ == '__main__':
    ged = GraphEditDistance()
    args = Arguments()
    # ged = GraphEditDistance()
    path = os.path.join(os.environ['DATA_ROOT'], args.split_type,
                        'heat_then_clean_then_slice/Apple-None-None-27/trial_T20220917_235349_019133')

    # if args.preprocess:
    #     preprocess_dataset(path, args.split_type)
    # test_set = CustomDataset(data_path=path)
    # # test_sampler = DistributedSampler(dataset=test_set, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    process(filepath=path, frames_per_segment=args.fp_seg)
    print('Done!')
