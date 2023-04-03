# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# Testing supervised model
import os
import sys
sys.path.append(os.environ['DATA_ROOT'])
sys.path.append(os.environ['BASELINES'])
from proScript.process_dataset import proScript_process
from proScript.proscript_args import Arguments
from proScript.utils import GraphEditDistance
from distributed_utils import *
from train_supervised import CustomDataset
from torchmetrics import MetricCollection
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def test_model(test_loader):  # validation
    with torch.no_grad():
        for inputs, outputs_nl, outputs_dsl in tqdm(test_loader, desc='Test'):
            inputs_tokenized = tokenizer(inputs, return_tensors="pt",
                                         padding="longest",
                                         max_length=max_source_length,
                                         truncation=True)
            # input_ids = tokenizer(inputs, return_tensors="pt").input_ids
            output_gen = t5_model.module.generate(inputs_tokenized["input_ids"].cuda(),
                                                  attention_mask=inputs_tokenized["attention_mask"].cuda(),
                                                  max_length=max_target_length,
                                                  do_sample=False)  # greedy generation
            output_pred = tokenizer.batch_decode(output_gen, skip_special_tokens=True)
            if args.output_type == 'nl':
                test_metrics.update(pred=output_pred, target=outputs_nl)
            elif args.output_type == 'dsl':
                test_metrics.update(pred=output_pred, target=outputs_dsl)
            # output_ids = tokenizer(outputs)
            # print('input: {} \n pred: {} \n true: {}'.format(inputs[ind], output_pred[ind], outputs[ind]))
        dist.barrier()
        test_ged = test_metrics['GraphEditDistance'].compute()
        dist.barrier()
        if is_main_process():
            print('Test GED: {}'.format(test_ged))
            log_file.write('Test GED: ' + str(test_ged.item()) + "\n")
            log_file.flush()


if __name__ == "__main__":
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
    ckpt_file = 'proscript_best_{}_{}.json'.format(args.output_type, str(args.run_id))
    proscript_ckpt_path = os.path.join(os.getcwd(), ckpt_file)
    test_filename = "proscript_test_{}.tsv".format(args.test_split)
    logger_filename = "proScript_log_{}.txt".format(args.run_id)
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        test_data_path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.test_split)
        if not os.path.isfile(test_filename):
            print("\n========= Processing Test Dataset {} ========\n".format(args.test_split))
            proScript_process(dir=test_data_path, data_filename=test_filename)
    # fixed based on train data
    max_source_length = 80
    max_target_length = 300

    test_dataset = CustomDataset(data_path='', data_filename=test_filename)
    print("Length of Test Dataset: {}\n".format(len(test_dataset)))
    test_sampler = DistributedSampler(dataset=test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                             num_workers=args.num_workers, shuffle=False, pin_memory=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(proscript_ckpt_path).cuda()
    t5_model = DDP(t5_model, device_ids=[local_rank])
    t5_model.eval()
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    metrics = MetricCollection([GraphEditDistance(dist_sync_on_step=True)]).cuda()
    test_metrics = metrics.clone(prefix='test_')
    test_model(test_loader)
    cleanup()
    log_file.close()
    print('Done!')
