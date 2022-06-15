#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 下午5:42
# @Author  : guoxz
# @Site    : 
# @File    : split_train_and_val.py
# @Software: PyCharm
# @Description

import codecs
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with codecs.open('../data/train.jsonl', 'r', encoding='utf8') as f:
        lines = [x.strip() for x in f.readlines()]

    train_data, val_data = train_test_split(lines, test_size=0.1, random_state=0)

    with codecs.open('../data/train.json', 'w', encoding='utf8') as f:
        f.writelines([x + '\n' for x in train_data])

    with codecs.open('../data/val.json', 'w', encoding='utf8') as f:
        f.writelines([x + '\n' for x in val_data])
