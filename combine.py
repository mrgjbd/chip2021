#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 下午2:36
# @Author  : guoxz
# @Site    :
# @File    : combine.py
# @Software: PyCharm
# @Description

import codecs
import json
import numpy as np
from tqdm import tqdm

idx2label = ['阴性', '阳性', '其他', '不标注']
label2idx = {x: i for i, x in enumerate(idx2label)}


def read_data(file):
    with codecs.open(file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]


# def combine2():
#     file1 = './data/result_longformer_mask_pooling_base1.txt'
#     file2 = './data/result_longformer_mask_pooling_lawformer.txt'
#
#     data1 = read_data(file1)
#     data2 = read_data(file2)
#
#     for d1, d2 in tqdm(zip(data1, data2), total=len(data1)):
#         assert d1['dialog_id'] == d2['dialog_id']
#         for dialog_data1, dialog_data2 in zip(d1['dialog_info'], d2['dialog_info']):
#             for ner1, ner2 in zip(dialog_data1['ner'], dialog_data2['ner']):
#                 assert ner1['mention'] == ner2['mention']
#                 prob1 = np.array(ner1['prob'])
#                 prob2 = np.array(ner2['prob'])
#                 prob = 0.5 * prob1 + 0.5 * prob2
#                 label = idx2label[np.argmax(prob)]
#
#                 # if ner1['attr'] != label:
#                 #     print()
#
#                 ner1['prob'] = prob.tolist()
#                 ner1['attr'] = label
#
#     with codecs.open('./data/result_longformer_mask_pooling_combine2.txt', 'w', encoding='utf8') as f:
#         for d in data1:
#             f.write(json.dumps(d, ensure_ascii=False) + '\n')


def combine3():
    file1 = './data/testb_base1.txt'
    file2 = './data/testb_lawformer.txt'
    file3 = './data/testb_base.txt'

    data1 = read_data(file1)
    data2 = read_data(file2)
    data3 = read_data(file3)

    for d1, d2, d3 in tqdm(zip(data1, data2, data3), total=len(data1)):
        assert d1['dialog_id'] == d2['dialog_id'] == d3['dialog_id']
        for dialog_data1, dialog_data2, dialog_data3 in zip(d1['dialog_info'], d2['dialog_info'], d3['dialog_info']):
            for ner1, ner2, ner3 in zip(dialog_data1['ner'], dialog_data2['ner'], dialog_data3['ner']):
                assert ner1['mention'] == ner2['mention'] == ner3['mention']
                prob1 = np.array(ner1['prob'])
                prob2 = np.array(ner2['prob'])
                prob3 = np.array(ner3['prob'])
                prob = 0.334 * prob1 + 0.333 * prob2 + 0.333 * prob3

                label = idx2label[np.argmax(prob)]

                # if ner1['attr'] != label:
                #     print()

                ner1['prob'] = prob.tolist()
                ner1['attr'] = label

    with codecs.open('./data/testb_combine3.txt', 'w', encoding='utf8') as f:
        for d in data1:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    combine3()
