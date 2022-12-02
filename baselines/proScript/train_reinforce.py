# training using REINFORCE
from process_dataset import proScript_process
from proscript_args import Arguments
from utils import GraphEditDistance
from ..distributed_utils import *
from torchmetrics import MetricCollection
from get_vocab import save_vocab
import os
import sys
import csv
import random
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.environ['DATA_ROOT'])


class CustomLoss():
    def __init__(self):
        super().__init__()
        self.m = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def batch_ll(self, logits, labels):
        batch_graph_prob = []
        batch_size = logits.shape[0]
        for sample_ind in range(batch_size):
            batch_graph_prob.append(-self.loss(self.m(logits[sample_ind].contiguous()), labels[sample_ind]))
        return torch.stack(batch_graph_prob)

    def compute_reward(self, pred_labels, outputs):
        output_pred = tokenizer.batch_decode(pred_labels, skip_special_tokens=True)
        metrics.update(pred=output_pred, target=outputs)
        batch_rewards = metrics['GraphEditDistance'].compute(reinforce=True)
        metrics.reset()
        return torch.tensor(batch_rewards, dtype=torch.float32).cuda()

    def compute_loss(self, input_ids, attention_mask, outputs):
        with torch.no_grad():
            t5_model.eval()
            pred_labels = t5_model.module.generate(input_ids.cuda(),
                                                   attention_mask=attention_mask.cuda(),
                                                   max_length=max_target_length,
                                                   do_sample=True,
                                                   bad_words_ids=bad_words_ids)  # greedy generation
        # forward pass
        t5_model.train()
        logits = t5_model(input_ids=input_ids.cuda(), labels=pred_labels.cuda()).logits  # [b, seq_len, vocab_size]
        # breakpoint()
        # compute custom loss
        # breakpoint()
        batch_log_lik = self.batch_ll(logits, pred_labels)  # [batch_size]
        batch_reward = self.compute_reward(pred_labels, outputs)  # [batch_size]
        loss = -torch.dot(batch_log_lik, batch_reward) / len(batch_log_lik)
        return loss, batch_reward.mean()


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
    train_rewards = []
    for inputs, outputs, _ in tqdm(train_loader, desc='Train'):
        input_tokenized = tokenizer(inputs,
                                   padding="longest",
                                   max_length=max_source_length,
                                   truncation=True,
                                   return_tensors="pt",
                                   add_special_tokens=False)
        input_ids, attention_mask = input_tokenized.input_ids, input_tokenized.attention_mask

        loss, rewards = custom_loss.compute_loss(input_ids, attention_mask, outputs)
        # loss = t5_model(input_ids=input_ids.cuda(), labels=true_labels.cuda()).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_rewards.append(rewards.item())
    dist.barrier()
    if is_main_process():
        print('Epoch: {} | Train Loss: {} | Train Rewards: {}'.format(epoch,
                                                                      np.array(train_loss).mean(),
                                                                      np.array(train_rewards).mean()))
        if epoch % 1 == 0:
            ind = random.randint(0, len(inputs)-1)
            with torch.no_grad():
                t5_model.eval()
                pred_labels = t5_model.module.generate(input_tokenized["input_ids"].cuda(),
                                                       max_length=max_target_length,
                                                       do_sample=False,
                                                       bad_words_ids=bad_words_ids)  # greedy generation
                # breakpoint()
                pred = tokenizer.decode(pred_labels[ind], skip_special_tokens=True)
                # breakpoint()
                print('input: {} \n pred: {} \n true: {}'.format(inputs[ind], pred, outputs[ind]))


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
    max_target_length = 130

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
    # t5_model = CustomTrainer(model=t5_model, train_dataset=)
    # transformers use layer norm (and not batch norm) which is local -- no need to sync across all instances
    tokenizer = T5Tokenizer.from_pretrained("t5-small", add_prefix_space=True)

    # limiting output vocab
    t5_vocab = list(tokenizer.get_vocab())
    if not os.path.isfile('my_vocab.json'):
        print("\n=========== Processing Vocab file for Transformer ==========\n")
        save_vocab(data_filename)
    my_vocab = json.load(open('my_vocab.json', 'r'))
    bad_words_ids = tokenizer([token for token in t5_vocab if token not in my_vocab]).input_ids

    optimizer = optim.AdamW(t5_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    custom_loss = CustomLoss()
    metrics = MetricCollection([GraphEditDistance()]).cuda()
    # train_metrics = metrics.clone(prefix='train_')
    # test_metrics = metrics.clone(prefix='test_')
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        train_epoch()

    cleanup()
    log_file.close()
    print('Done!')
