#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 上午11:21
# @Author  : guoxz
# @Site    : 
# @File    : datasets.py
# @Software: PyCharm
# @Description

import codecs
import itertools
import json
import random
from copy import deepcopy

import torch
from tqdm import tqdm
from collections import defaultdict


class Dataset(torch.utils.data.Dataset):
    PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

    def __init__(self, file, tokenizer=None, max_len=5120, for_train=True, enhance=False, random_cut=False):

        self.max_len = max_len
        self.for_train = for_train
        self.enhance = enhance
        self.tokenizer = tokenizer
        self.idx2label = ['阴性', '阳性', '其他', '不标注']
        self.label2idx = {x: i for i, x in enumerate(self.idx2label)}
        self.file = file
        self.all_data = None
        self.load(random_cut)

    def _words_not_in_sent(self, sent):
        for word in ['隐私问题无法', '自动回复', '尽快回复']:
            if word in sent:
                return False
        return True

    def _enhance_dialog(self, dialog):
        text = dialog['text']
        ners = deepcopy(dialog['ner'])
        if len(ners) > 0:
            ners = sorted(ners, key=lambda a: a['range'][0] * 10000 + a['range'][1])
            # 确定实体之间不交叉
            ner_pos = list(
                itertools.chain.from_iterable([list(range(ner['range'][0], ner['range'][1])) for ner in ners]))
            if len(ner_pos) != len(set(ner_pos)):
                return

            new_text = ''
            start = 0
            end = 0
            for ner in ners:
                end = ner['range'][0]
                new_text += text[start:end]
                start = ner['range'][1]

                mention = ner['mention']
                name = ner['name']

                if random.randint(0, 5) == 1:
                    new_mention = name if name != 'undefined' and random.randint(0, 1) == 0 else mention
                else:
                    new_mention = []
                    for char in (name if name != 'undefined' and random.randint(0, 1) == 0 else mention):
                        rnd_state = random.uniform(0, 1)
                        if rnd_state < 0.05:
                            new_mention.append(char + random.choice(self.PUNCTUATIONS))
                        elif rnd_state < 0.1:
                            new_mention.append(char + char)
                        else:
                            new_mention.append(char)
                    new_mention = ''.join(new_mention)

                ner['range'][0] = len(new_text)
                ner['range'][1] = len(new_text) + len(new_mention)

                new_text += new_mention
                ner['mention'] = new_mention

            new_text += text[start:]

            dialog['text'] = new_text
            dialog['ner'] = ners

    def load(self, random_cut=False):
        print(f'load data, random_cut: {random_cut}')

        freq = defaultdict(int)

        all_data = []
        with codecs.open(self.file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for ii, line in enumerate(lines):
                d = json.loads(line)
                dialog_info = d['dialog_info']

                # check
                for dialog in dialog_info:
                    text = dialog['text']
                    for ner in dialog['ner']:
                        mention = ner['mention']
                        assert mention == text[ner['range'][0]:ner['range'][1]]
                        freq[ner['attr']] += 1

                dialog_info = [x for x in dialog_info if self._words_not_in_sent(x['text']) or len(x['ner']) > 0]
                if self.for_train and self.enhance:
                    for dialog in dialog_info:
                        self._enhance_dialog(dialog)

                if random_cut:
                    dialog_info_with_ner = [x for x in dialog_info if x['ner']]
                    dialog_info_without_ner = [x for x in dialog_info if not x['ner']]
                    if len(dialog_info_with_ner) > 3:
                        new_dialog_info_with_ner = [x for x in dialog_info_with_ner if random.uniform(0, 1) > 0.25]
                        if len(new_dialog_info_with_ner) == 0:
                            new_dialog_info_with_ner = dialog_info_with_ner
                        new_dialog_info_without_ner = [x for x in dialog_info_without_ner if
                                                       random.uniform(0, 1) > 0.25]

                        dialog_info = sorted(new_dialog_info_with_ner + new_dialog_info_without_ner,
                                             key=lambda x: int(x['sentence_id']))

                token_cache = ['[XLS]']
                token_type_ids = [0]
                start_info = []
                end_info = []
                labels = []
                for x in dialog_info:
                    text = x['text'].lower()
                    sender = '[DOCOTOR]' if x['sender'] == '医生' else '[PATIEND]'
                    token_cache.append(sender)
                    token_type_id = 1 if sender == '[DOCOTOR]' else 2
                    x['ner'] = sorted(x['ner'], key=lambda a: a['range'][0] * 10000 + a['range'][1])
                    if self.for_train:
                        for ner in x['ner']:
                            if ner['attr'] not in self.idx2label:
                                continue
                            start = ner['range'][0] + len(token_cache)
                            end = ner['range'][1] + len(token_cache) - 1
                            start_info.append(start)
                            end_info.append(end)
                            labels.append(self.label2idx[ner['attr']])
                        token_cache.extend(list(text))
                        # token_cache.append('[SEP]')
                        token_type_ids.extend([token_type_id] * (len(text) + 1))
                    else:
                        for ner in x['ner']:
                            start = ner['range'][0] + len(token_cache)
                            end = ner['range'][1] + len(token_cache) - 1
                            start_info.append(start)
                            end_info.append(end)
                        token_cache.extend(list(text))
                        # token_cache.append('[SEP]')
                        token_type_ids.extend([token_type_id] * (len(text) + 1))
                if self.for_train:
                    all_data.append((token_cache + ['[SEP]'], start_info, end_info, token_type_ids + [0], labels))
                else:
                    all_data.append((token_cache + ['[SEP]'], start_info, end_info, token_type_ids + [0]))

        print(f'max len: {max([len(x[0]) for x in all_data])}')
        print(freq)
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):

        if self.for_train:
            tokens, start_info, end_info, token_type_ids, labels = self.all_data[item]
            entity_ids = [0] * len(tokens)
            for s, e in zip(start_info, end_info):
                entity_ids[s:e + 1] = [1] * (e - s + 1)
                entity_ids[s] = 2
                entity_ids[e] = 2
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, token_type_ids, start_info, end_info, entity_ids, labels
        else:
            tokens, start_info, end_info, token_type_ids = self.all_data[item]
            entity_ids = [0] * len(tokens)
            for s, e in zip(start_info, end_info):
                entity_ids[s:e + 1] = [1] * (e - s + 1)
                entity_ids[s] = 2
                entity_ids[e] = 2
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return input_ids, token_type_ids, start_info, end_info, entity_ids


if __name__ == '__main__':

    from transformers import BertTokenizer

    vocab_file = '../data/vocab.txt'
    tokenizer = BertTokenizer(vocab_file=vocab_file,
                              do_lower_case=True)
    file = '../data/all.json'
    dataset = Dataset(file, tokenizer=tokenizer, for_train=True)

    for i, x in tqdm(enumerate(dataset), total=len(dataset)):
        # print(i)
        s = x

    # tokenizer.cls_token_id
