# SUPERVISED TRAINING
# sudo apt install graphviz
# pip install graphviz
# pip install pydot
# pip install sentencepiece
# pip install nltk
# import nltk
# nltk.download('punkt')
from process_dataset import proScript_process
from proscript_args import Arguments
from proscript_utils import GraphEditDistance
from ..distributed_utils import *
from torchmetrics import MetricCollection
import os
import sys
import csv
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.environ['DATA_ROOT'])


class CustomDataset(Dataset):
    def __init__(self, data_path, data_filename):
        self.dataset_tsv_path = os.path.join(data_path, data_filename)
        self.file_tuples = [tuple(sample[0].split('\t')) for sample in list(csv.reader(open(self.dataset_tsv_path),
                                                                                       delimiter="\n"))]

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, item):
        return self.file_tuples[item]  # tuple: (input_hypothesis, output_graph_DOT, output_dsl_graph_DOT)


def train_epoch():
    t5_model.train()
    train_loss = []
    for inputs, _, outputs in tqdm(train_loader, desc='Train'):
        input_encoding = tokenizer(inputs,
                                   padding="longest",
                                   max_length=max_source_length,
                                   truncation=True,
                                   return_tensors="pt")
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        output_encodings = tokenizer(outputs,
                                     padding="longest",
                                     max_length=max_target_length,
                                     truncation=True,
                                     return_tensors="pt")
        labels = output_encodings.input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        loss = t5_model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), labels=labels.cuda()).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    dist.barrier()
    if epoch % 2 == 0:
        test_ged = test()
        dist.barrier()
        if is_main_process():
            print('Epoch: {} | Train Loss: {} | Test GED: {}'.format(epoch, np.array(train_loss).mean(), test_ged))


def test():
    t5_model.eval()
    with torch.no_grad():
        for inputs, _, outputs in tqdm(test_loader, desc='Test'):
            inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True)
            # input_ids = tokenizer(inputs, return_tensors="pt").input_ids
            output_gen = t5_model.module.generate(inputs_tokenized["input_ids"].cuda(),
                                                  attention_mask=inputs_tokenized["attention_mask"].cuda(),
                                                  max_length=max_target_length,
                                                  do_sample=False)  # greedy generation
            output_pred = tokenizer.batch_decode(output_gen, skip_special_tokens=True)
            test_metrics.update(pred=output_pred, target=outputs)
            # output_ids = tokenizer(outputs)
            ind = random.randint(0, len(inputs)-1)
            print('input: {} \n pred: {} \n true: {}'.format(inputs[ind], output_pred[ind], outputs[ind]))
    return test_metrics['GraphEditDistance'].compute()


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
    data_filename = "proscript_data.tsv"
    test_filename = "proscript_test_{}.tsv".format(args.test_split)
    logger_filename = "proScript_log_{}.txt".format(args.run_id)
    logger_path = os.path.join(os.getcwd(), logger_filename)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    if args.preprocess:
        train_data_path = os.path.join(os.environ['DATA_ROOT'], 'train')
        test_data_path = os.path.join(os.environ['DATA_ROOT'], 'test_splits', args.test_split)
        if not os.path.isfile(data_filename):
            print("\n========= Processing Train Dataset ========\n")
            max_source_length, max_target_length = \
                proScript_process(dir=train_data_path, data_filename=data_filename)
            print('\nmax_source_length: {} | max_target_length: {}\n'.format(max_source_length, max_target_length))
        if not os.path.isfile(test_filename):
            print("\n========= Processing Test Dataset {} ========\n".format(args.test_split))
            proScript_process(dir=test_data_path, data_filename=test_filename)
    # fixed based on train data
    max_source_length = 50
    max_target_length = 150

    dataset = CustomDataset(data_path='', data_filename=data_filename)
    test_dataset = CustomDataset(data_path='', data_filename=test_filename)
    print("Length of Train Dataset: {}".format(len(dataset)))  # 5363 samples (only positive entailed text)
    print("Length of Test Dataset: {}\n".format(len(test_dataset)))
    # train_size = int(args.data_split * len(dataset))
    # val_size = len(dataset) - train_size
    # train_set, val_set = random_split(dataset, [train_size, val_size])
    train_sampler, test_sampler = DistributedSampler(dataset=dataset, shuffle=True), \
                                  DistributedSampler(dataset=test_dataset, shuffle=True)
    train_loader, test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                           num_workers=args.num_workers, pin_memory=True), \
                                DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                           num_workers=args.num_workers, shuffle=False, pin_memory=True)

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()
    t5_model = DDP(t5_model, device_ids=[local_rank])
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    optimizer = optim.AdamW(t5_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics = MetricCollection([GraphEditDistance(dist_sync_on_step=True)]).cuda()
    # train_metrics = metrics.clone(prefix='train_')
    test_metrics = metrics.clone(prefix='test_')
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        train_epoch()
        # train_metrics.reset()
        test_metrics.reset()
    cleanup()
    log_file.close()
    print('Done!')
