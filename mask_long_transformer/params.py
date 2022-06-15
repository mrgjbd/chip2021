#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 下午2:35
# @Author  : guoxz
# @Site    : 
# @File    : params.py
# @Software: PyCharm
# @Description

import argparse


def construct_hyper_param():
    parser = argparse.ArgumentParser()
    # train settings
    parser.add_argument('--num_train_epochs', default=15, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size")
    parser.add_argument("--update_freq", default=1, type=int,
                        help="update_freq")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help='model learning rate.')

    # bert settings
    parser.add_argument("--bert_config_file",
                        default='/opt/disk2/nlp/预训练语言模型/longformer/base1/config.json',
                        type=str,
                        help="bert_config_file")
    parser.add_argument("--bert_vocab_file",
                        default='../data/vocab.txt',
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_init_checkpoint",
                        default='/opt/disk2/nlp/预训练语言模型/longformer/base1/pytorch_model.bin',
                        type=str,
                        help="bert_init_checkpoint")
    parser.add_argument("--save_dir",
                        default='../saves/base1',
                        type=str,
                        help="save_dir")

    parser.add_argument("--n_cross_thought",
                        help="n_cross_thought",
                        type=int,
                        default=5)

    parser.add_argument("--n_cross_thought_layers",
                        help="n_cross_thought_layers",
                        type=int,
                        default=3)

    parser.add_argument("--with_adv_train",
                        action='store_true',
                        help="with_adv_train")

    parser.add_argument("--data_dir",
                        default='./data', type=str,
                        help="save_dir")

    args = parser.parse_args()

    return args


args = construct_hyper_param()
