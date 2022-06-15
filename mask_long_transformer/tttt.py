#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2021/10/15 上午9:56 
# @Author   : guoxz
# @Site     : 
# @File     : tttt.py 
# @Software : PyCharm
# @Desc     :

import codecs
import json
import os

from recordclass import recordclass

NER_Sample = recordclass("NER_Sample", "ner_type start end string ner_classification")


def get_all_txt_file(base_dir, read_empty=False):
    all_txt_files = []
    for root, dirs, files in os.walk(base_dir):
        all_txt_files.extend([os.path.join(root, x) for x in files if x.endswith('.txt')])

    if read_empty:
        return [x for x in all_txt_files if
                os.path.exists(x.replace('.txt', '.ann')) and read_file(x.replace('.txt', '.ann')) == '']
    return [x for x in all_txt_files if
            os.path.exists(x.replace('.txt', '.ann')) and read_file(x.replace('.txt', '.ann')) != '']


def read_file(file):
    with codecs.open(file, 'r', encoding='utf8') as f:
        return ''.join(f.readlines())


def read_ann(ann_file):
    data = []

    with codecs.open(ann_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            tokens = line.strip().split('\t')
            if tokens[0].startswith('T') and len(tokens) == 6:
                data.append(NER_Sample(
                    ner_type=tokens[1],
                    start=int(tokens[2]) - 1,
                    end=int(tokens[3]) - 1,
                    string=tokens[4],
                    ner_classification=tokens[5]
                ))
    return data


def get_all_data(base_dir='/opt/disk2/nlp/华西/正式标注内容', lower_case=True, filter_re_empty=True):
    all_txt_files = get_all_txt_file(base_dir)
    all_data = []
    for txt_file in all_txt_files:
        ann_data = []
        text = read_file(txt_file)
        if lower_case:
            text = text.lower()
        data = read_ann(txt_file.replace('.txt', '.ann'))

    return text, data


def get_all_ner_data(base_dir='/opt/disk2/nlp/华西/正式标注内容', lower_case=True):
    all_txt_files = get_all_txt_file(base_dir)
    all_data = []
    for txt_file in all_txt_files:
        text = read_file(txt_file)
        if lower_case:
            text = text.lower()
        data = read_ann(txt_file.replace('.txt', '.ann'))

        all_data.append((txt_file, text, data))
    return all_data


if __name__ == '__main__':
    all_data = get_all_ner_data('/opt/disk2/nlp/华西/0617(截止0930)超声报告标注结果_denoise')
    ret = []
    for i, (txt_file, text, data) in enumerate(all_data):
        dialog_id = i
        ner = []
        for d in data:
            start = d['start']
            end = d['end']
            mention = d['string']
            name = d['ner_type']
            attr = d['ner_classification']
            if name == '部位' and attr in ['阴性', '阳性']:
                range = [start, end + 1]
                ner.append({
                    'name': name,
                    'mention': mention,
                    'range': range,
                    'type': 'type',
                    'attr': attr
                })

        ret.append({
            'dialog_id': dialog_id,
            'dialog_info': [{
                'text': text,
                'sentence_id': 1,
                'ner': ner,
                'sender': '患者'
            }]
        })

    with codecs.open('../data/pretrain.json', 'w', encoding='utf8') as f:
        for x in ret:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
