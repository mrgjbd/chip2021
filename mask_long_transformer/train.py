#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2021/10/15 上午11:34 
# @Author   : guoxz
# @Site     : 
# @File     : train.py 
# @Software : PyCharm
# @Desc     :
import os
import random
from copy import deepcopy
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import LongformerConfig, LongformerModel

from adv import FGM
from mask_long_transformer.data_parallel import DataParallelImbalance
from mask_long_transformer.datasets import Dataset
from mask_long_transformer.model import CHIP2021Model
from mask_long_transformer.params import args
from optimization import BertAdam


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 1e-6)


def load_pretrained_bert(bert_model, init_checkpoint):
    if init_checkpoint is not None:
        state = torch.load(init_checkpoint, map_location='cpu')
        if 'model_bert_best' in init_checkpoint:
            bert_model.load_state_dict(state['model_bert'], strict=False)
        else:
            state = {k.replace('bert.', ''): v for k, v in state.items() if not k.startswith('cls.')}
            state['embeddings.token_type_embeddings.weight'] = state['embeddings.token_type_embeddings.weight'][:2, :]
            bert_model.load_state_dict(state, strict=True)


def get_model_input(data):
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    # input_ids, token_type_ids, start_info, end_info, entity_ids, labels

    bs = len(data)
    max_len = max([len(x[0]) for x in data])
    n_max_pos = max([len(x[1]) for x in data])
    max_n_entity = max([len(x[-1]) for x in data])

    input_ids_list = []
    att_mask_list = []
    global_attention_mask_list = []
    position_ids_list = []
    entity_ids_list = []
    token_type_ids_list = []
    start_list = []
    end_list = []
    attrs_list = []

    addition_input_ids = np.zeros((bs, max_n_entity), dtype=np.int64)
    addition_pos_ids = np.zeros((bs, max_n_entity), dtype=np.int64)
    addition_att_mask = np.zeros((bs, max_n_entity, max_len), dtype=np.int64)

    for i, d in enumerate(data):
        input_ids, token_type_ids, start_info, end_info, entity_ids, labels = deepcopy(d)
        len_labels = len(labels)

        attention_mask = [1] * len(input_ids)
        input_ids = pad(input_ids, max_len, 0)
        attention_mask = pad(attention_mask, max_len, 0)
        entity_ids = pad(entity_ids, max_len, 0)
        token_type_ids = pad(token_type_ids, max_len, 0)

        input_ids_list.append(input_ids)
        att_mask_list.append(attention_mask)
        entity_ids_list.append(entity_ids)
        token_type_ids_list.append(token_type_ids)

        pos_start = random.randint(0, 5120 - max_len)

        position_ids = list(range(pos_start, pos_start + max_len))
        position_ids[0] = 0
        position_ids_list.append(position_ids)

        global_attention_mask = np.array([0] * max_len, dtype=np.int64)
        global_attention_mask[start_info] = 1
        # global_attention_mask[end_info] = 1
        global_attention_mask[0] = 1
        global_attention_mask_list.append(global_attention_mask)

        start_info = pad(start_info, max_n_entity, -1)
        end_info = pad(end_info, max_n_entity, -1)
        labels = pad(labels, max_n_entity, -1)

        start_list.append(start_info)
        end_list.append(end_info)
        attrs_list.append(labels)

        addition_input_ids[i, ...] = bert_tokenizer.mask_token_id
        # addition_input_ids[i, :len_labels] = bert_tokenizer.mask_token_id
        addition_pos_ids[i, :len_labels] = np.array([s + pos_start for s in start_info[:len_labels]], dtype=np.int64)

        for entity_idx, (s, e) in enumerate(zip(start_info[:len_labels], end_info[:len_labels])):
            addition_att_mask[i, entity_idx, s:e + 1] = 1

    input_ids = np.array(input_ids_list, dtype=np.int64)
    attention_mask = np.array(att_mask_list, dtype=np.int64)
    token_type_ids = np.array(token_type_ids_list, dtype=np.int64)
    entity_ids = np.array(entity_ids_list, dtype=np.int64)
    global_attention_mask = np.array(global_attention_mask_list)
    position_ids = np.array(position_ids_list, dtype=np.int64)
    start = np.array(start_list, dtype=np.int64)
    end = np.array(end_list, dtype=np.int64)
    labels = np.array(attrs_list, dtype=np.int64)

    new_addition_att_mask = np.zeros_like(addition_att_mask, dtype=np.float32)
    new_addition_att_mask[addition_att_mask == 0] = -10000  # float('-inf')
    new_addition_att_mask[addition_att_mask == 1] = 0

    input_ids = torch.from_numpy(input_ids).to(device)
    attention_mask = torch.from_numpy(attention_mask).to(device)

    global_attention_mask = torch.from_numpy(global_attention_mask).to(device)
    position_ids = torch.from_numpy(position_ids).to(device)
    start = torch.from_numpy(start).to(device)
    end = torch.from_numpy(end).to(device)
    labels = torch.from_numpy(labels).to(device)
    token_type_ids = torch.from_numpy(token_type_ids).to(device)
    entity_ids = torch.from_numpy(entity_ids).to(device)

    addition_input_ids = torch.from_numpy(addition_input_ids).to(device)
    addition_pos_ids = torch.from_numpy(addition_pos_ids).to(device)
    addition_att_mask = torch.from_numpy(new_addition_att_mask).to(device)

    return input_ids, token_type_ids, attention_mask, global_attention_mask, position_ids, entity_ids, \
           start, end, labels, addition_input_ids, addition_pos_ids, addition_att_mask


def batch_index(input, index):
    dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), input.size(2))
    out = torch.gather(input, 1, dummy)
    return out


def train(model, bert_tokenizer, train_loader, val_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = int(len(train_loader) * args.num_train_epochs /
                  args.accumulate_gradients)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    adv = FGM(model) if args.with_adv_train else None

    best_f1 = 0.0
    best_acc = 0.0
    global_step = 0
    for i_epoch in range(1, 1 + args.num_train_epochs):

        total_loss = 0.0
        total_adv_loss = 0.0

        iter_bar = tqdm(train_loader, total=len(train_loader), desc=f'epoch {i_epoch}, train...')
        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        for step, batch in enumerate(iter_bar):
            global_step += 1

            input_ids, token_type_ids, attention_mask, global_attention_mask, position_ids, entity_ids, \
            start, end, labels, addition_input_ids, addition_pos_ids, addition_att_mask = batch

            logits, loss = model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 global_attention_mask=global_attention_mask,
                                 position_ids=position_ids,
                                 start_pos=start,
                                 end_pos=end,
                                 attrs=labels,
                                 addition_input_ids=addition_input_ids,
                                 addition_pos_ids=addition_pos_ids,
                                 addition_att_mask=addition_att_mask,
                                 for_train=True)
            loss = loss.mean()
            loss.backward()
            total_loss += loss.item()

            if adv is not None and i_epoch > 2:
                adv.attack()
                adv_logits, adv_loss = model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             global_attention_mask=global_attention_mask,
                                             position_ids=position_ids,
                                             start_pos=start,
                                             end_pos=end,
                                             attrs=labels,
                                             addition_input_ids=addition_input_ids,
                                             addition_pos_ids=addition_pos_ids,
                                             addition_att_mask=addition_att_mask,
                                             for_train=True)
                adv_loss = adv_loss.mean()
                adv_loss.backward()
                adv.restore()
                total_adv_loss += adv_loss.item()

            if global_step % 50 == 0:
                if adv is None or i_epoch <= 2:
                    print(f'loss: {total_loss / (step + 1)}')
                else:
                    print(f'loss: {total_loss / (step + 1)}, adv loss: {total_adv_loss / (step + 1)}')

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if (step + 1) % args.accumulate_gradients == 0:
                lr_this_step = args.lr * \
                               warmup_linear(global_step / t_total,
                                             args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()

        # if i_epoch < int(args.num_train_epochs * 0.7):
        #     train_loader.dataset.load(random_cut=True)
        # elif i_epoch == int(args.num_train_epochs * 0.7):
        #     train_loader.dataset.load(random_cut=False)

        # train_loader.dataset.load()

        iter_bar = tqdm(val_loader, total=len(val_loader), desc='val')
        model.eval()
        ret_info = {}
        for s in val_dataset.idx2label:
            ret_info[s] = {'total_num': 0,
                           'pred_num': 0,
                           'correct': 0}
        with torch.no_grad():
            # eva_info = []
            t_total = 0
            t_correct = 0
            for batch in iter_bar:
                input_ids, token_type_ids, attention_mask, global_attention_mask, position_ids, entity_ids, \
                start, end, labels, addition_input_ids, addition_pos_ids, addition_att_mask = batch

                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               global_attention_mask=global_attention_mask,
                               position_ids=position_ids,
                               start_pos=start,
                               end_pos=end,
                               attrs=labels,
                               addition_input_ids=addition_input_ids,
                               addition_pos_ids=addition_pos_ids,
                               addition_att_mask=addition_att_mask,
                               for_train=False)

                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                labels = labels.view(-1)
                t_correct += (pred[labels != -1] == labels[labels != -1]).detach().cpu().numpy().sum()
                t_total += len(labels[labels != -1])

                for pred_label, true_label in zip(pred.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                    if true_label == -1:
                        continue
                    ret_info[val_dataset.idx2label[pred_label]]['pred_num'] += 1
                    ret_info[val_dataset.idx2label[true_label]]['total_num'] += 1
                    if pred_label == true_label:
                        ret_info[val_dataset.idx2label[true_label]]['correct'] += 1

            acc = t_correct / t_total

            val_f1 = 0.0
            for s in val_dataset.idx2label:
                p = ret_info[s]['correct'] / (ret_info[s]['pred_num'] + 1e-5)
                r = ret_info[s]['correct'] / (ret_info[s]['total_num'] + 1e-5)
                f1 = 2 * p * r / (p + r + 1e-5)
                val_f1 += f1
                ret_info[s]['f1'] = f1

            val_f1 /= len(val_dataset.idx2label)
            print()
            print(f'acc: {acc}')
            if acc > best_acc:
                print(f'saved! new best acc {acc}, ori_acc {best_acc}')
                model_to_save = model.module if hasattr(model, 'module') else model
                os.makedirs('../saves', exist_ok=True)
                torch.save(model_to_save.state_dict(), os.path.join(args.save_dir, 'model.bin'))
                best_acc = acc
            else:
                print(f'current acc: {acc}, best acc: {best_acc}')
            pprint(ret_info)
            print()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    n_labels = 4
    n_gpu = torch.cuda.device_count()
    args.with_adv_train = True
    args.batch_size = 2

    if n_gpu > 0:
        torch.cuda.manual_seed_all(0)
        # random.seed(0)
        np.random.seed(0)

    vocab_file = args.bert_vocab_file
    bert_tokenizer = BertTokenizer(vocab_file=vocab_file,
                                   do_lower_case=True)

    train_dataset = Dataset('../data/all.json',
                            tokenizer=bert_tokenizer,
                            enhance=False)

    val_dataset = Dataset('../data/val.json',
                          tokenizer=bert_tokenizer,
                          enhance=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    longformerconfig = LongformerConfig.from_json_file(args.bert_config_file)
    longformerconfig.max_position_embeddings = 4096 + 1024
    longformermodel = LongformerModel(config=longformerconfig)

    # args.bert_init_checkpoint='../saves/longformer_base1_pretrain_model_12000.bin'
    state = torch.load(args.bert_init_checkpoint, map_location='cpu')
    state = {k.replace('roberta.', '').replace('bert.', '').replace('longformer.', ''): v for k, v in state.items()}
    state['embeddings.position_embeddings.weight'] = torch.cat(
        [state['embeddings.position_embeddings.weight'],
         state['embeddings.position_embeddings.weight'][-(
                 longformerconfig.max_position_embeddings - state['embeddings.position_embeddings.weight'].shape[0]):]],
        dim=0)

    if 'embeddings.position_ids' in state:
        del state['embeddings.position_ids']

    longformermodel.load_state_dict(state, strict=False)

    model = CHIP2021Model(longformerconfig, longformermodel, n_labels=n_labels)
    model = model.to(device)
    train_sampler = RandomSampler(train_dataset, replacement=False)

    if n_gpu > 1:
        model = DataParallelImbalance(model)

    train_loader = torch.utils.data.DataLoader(
        batch_size=args.batch_size,
        dataset=train_dataset,
        shuffle=True,
        num_workers=0,
        collate_fn=get_model_input,
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        batch_size=args.batch_size,
        dataset=val_dataset,
        shuffle=False,
        num_workers=0,
        collate_fn=get_model_input,
        drop_last=False
    )

    train(model, bert_tokenizer, train_loader, val_loader)
