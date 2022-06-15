#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2021/10/15 上午11:36 
# @Author   : guoxz
# @Site     : 
# @File     : model.py 
# @Software : PyCharm
# @Desc     :

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction,
                                            ignore_index=self.ignore_index)
        return loss


class CHIP2021Model(nn.Module):
    def __init__(self, config, longformer_model, n_labels=4):
        super(CHIP2021Model, self).__init__()

        self.config = config
        self.n_labels = n_labels
        self.longformer_model = longformer_model

        self.query = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.key = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.value = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.output = nn.Linear(config.hidden_size, self.n_labels)

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            global_attention_mask=None,
            start_pos=None,
            end_pos=None,
            attrs=None,
            weight=None,
            addition_input_ids=None,
            addition_pos_ids=None,
            addition_att_mask=None,
            for_train=True,
            use_r_drop=False):

        bs, length = input_ids.size()
        if use_r_drop:
            assert bs == 2
            bert_output = self.longformer_model(input_ids=input_ids,
                                                attention_mask=attention_mask.repeat((bs, 1)),
                                                global_attention_mask=global_attention_mask.repeat((bs, 1)),
                                                position_ids=position_ids.repeat((bs, 1)))

            sequence_output = bert_output.last_hidden_state
            sequence_output1 = sequence_output[0]
            sequence_output2 = sequence_output[1]

            start_pos = start_pos[start_pos != -1]
            end_pos = end_pos[end_pos != -1]

            start_states1 = sequence_output1[start_pos]
            end_state1 = sequence_output1[end_pos]
            entity_states1 = torch.cat([start_states1, end_state1], dim=-1)
            entity_states1 = entity_states1.contiguous()
            logits1 = self.entity_output_layer(entity_states1)

            start_states2 = sequence_output2[start_pos]
            end_state2 = sequence_output2[end_pos]
            entity_states2 = torch.cat([start_states2, end_state2], dim=-1)
            entity_states2 = entity_states2.contiguous()
            logits2 = self.entity_output_layer(entity_states2)

            ce = nn.CrossEntropyLoss()
            ce_loss = (ce(logits1, attrs[attrs != -1]) + ce(logits2, attrs[attrs != -1])) / 2
            kl_loss = self.compute_kl_loss(logits1, logits2)
            return ce_loss, kl_loss

        bert_output = self.longformer_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            global_attention_mask=global_attention_mask,
                                            position_ids=position_ids)

        sequence_output = bert_output.last_hidden_state

        addition_hidden_states = self.longformer_model.embeddings(input_ids=addition_input_ids,
                                                                  position_ids=addition_pos_ids)

        key = self.key(sequence_output)
        value = self.value(sequence_output)
        query = self.query(addition_hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores /= math.sqrt(query.size()[-1])
        attention_scores = attention_scores + addition_att_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.att_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        x = self.dense(context)
        x = gelu(x)
        x = self.layer_norm(x)
        prediction_scores = self.output(x)

        if for_train:
            logits = prediction_scores.view((-1, self.n_labels))
            attrs = attrs.view(-1)
            # ['阴性', '阳性', '其他', '不标注']
            #  14086   74774  6167   23949
            loss = nn.CrossEntropyLoss(ignore_index=-1)(logits, attrs)
            return logits, loss
        else:
            logits = prediction_scores.view((-1, self.n_labels))
            return logits
