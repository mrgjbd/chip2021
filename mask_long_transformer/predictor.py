#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time     : 2021/10/15 下午5:11 
# @Author   : guoxz
# @Site     : 
# @File     : predictor.py 
# @Software : PyCharm
# @Desc     :

import codecs
import itertools
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, LongformerConfig, LongformerModel

from mask_long_transformer.datasets import Dataset
from mask_long_transformer.model import CHIP2021Model


class Predictor(object):
    def __init__(self, test_file, vocab_file, config, init_checkpoint,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.bert_tokenizer = BertTokenizer(vocab_file=vocab_file,
                                            do_lower_case=True)

        longformerconfig = LongformerConfig.from_json_file(config)
        longformerconfig.max_position_embeddings = 5120
        # longformerconfig.attention_window=[512+128]*12
        longformermodel = LongformerModel(config=longformerconfig)
        model = CHIP2021Model(longformerconfig, longformermodel)

        state = torch.load(init_checkpoint, map_location='cpu')
        # state = {k.replace('longformer_model.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()
        self.idx2label = ['阴性', '阳性', '其他', '不标注']
        self.label2idx = {x: i for i, x in enumerate(self.idx2label)}

        self.dataset = Dataset(test_file, tokenizer=self.bert_tokenizer, for_train=False)

        self.ori_data = []
        with codecs.open(test_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                d = json.loads(line)
                dialog_info = d['dialog_info']
                for x in dialog_info:
                    x['ner'] = sorted(x['ner'], key=lambda a: a['range'][0] * 10000 + a['range'][1])
                self.ori_data.append(d)

    def _get_model_input(self, data):
        input_ids, token_type_ids, start_info, end_info, entity_ids = data

        length = len(input_ids)
        n_entity = len(start_info)

        attention_mask = [1] * len(input_ids)
        global_attention_mask = [0] * len(input_ids)
        global_attention_mask[0] = 1
        position_ids = list(range(0, len(input_ids)))

        input_ids = np.array([input_ids], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.int64)
        global_attention_mask = np.array([global_attention_mask])
        token_type_ids = np.array([token_type_ids])
        entity_ids = np.array([entity_ids])
        position_ids = np.array([position_ids], dtype=np.int64)
        start = np.array([start_info], dtype=np.int64)
        end = np.array([end_info], dtype=np.int64)
        global_attention_mask[:, start[0]] = 1
        # global_attention_mask[:, end[0]] = 1

        addition_input_ids = np.array([[self.bert_tokenizer.mask_token_id] * n_entity], dtype=np.int64)
        addition_pos_ids = np.array([start_info], dtype=np.int64)
        addition_att_mask = np.zeros((1, n_entity, length), dtype=np.int64)
        for entity_idx, (s, e) in enumerate(zip(start_info, end_info)):
            addition_att_mask[0, entity_idx, s:e + 1] = 1

        new_addition_att_mask = np.zeros_like(addition_att_mask, dtype=np.float32)
        new_addition_att_mask[addition_att_mask == 0] = -10000  # float('-inf')
        new_addition_att_mask[addition_att_mask == 1] = 0
        addition_att_mask = new_addition_att_mask

        input_ids = torch.from_numpy(input_ids).to(self.device)
        attention_mask = torch.from_numpy(attention_mask).to(self.device)
        global_attention_mask = torch.from_numpy(global_attention_mask).to(self.device)
        token_type_ids = torch.from_numpy(token_type_ids).to(self.device)
        entity_ids = torch.from_numpy(entity_ids).to(self.device)
        position_ids = torch.from_numpy(position_ids).to(self.device)
        start = torch.from_numpy(start).to(self.device)
        end = torch.from_numpy(end).to(self.device)

        addition_input_ids = torch.from_numpy(addition_input_ids).to(self.device)
        addition_pos_ids = torch.from_numpy(addition_pos_ids).to(self.device)
        addition_att_mask = torch.from_numpy(addition_att_mask).to(self.device)

        return input_ids, token_type_ids, attention_mask, global_attention_mask, position_ids, \
               entity_ids, start, end, addition_input_ids, addition_pos_ids, addition_att_mask

    def pred(self, target_file):
        with torch.no_grad():
            for i, x in tqdm(enumerate(self.dataset), total=len(self.dataset)):
                input_ids, token_type_ids, attention_mask, global_attention_mask, position_ids, entity_ids, \
                start, end, addition_input_ids, addition_pos_ids, addition_att_mask = self._get_model_input(x)

                logits = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    global_attention_mask=global_attention_mask,
                                    position_ids=position_ids,
                                    start_pos=start,
                                    end_pos=end,
                                    attrs=None,
                                    addition_input_ids=addition_input_ids,
                                    addition_pos_ids=addition_pos_ids,
                                    addition_att_mask=addition_att_mask,
                                    for_train=False)

                prob = torch.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1)

                prob = prob.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()

                d = self.ori_data[i]
                ners = list(itertools.chain.from_iterable([a['ner'] for a in d['dialog_info']]))
                assert len(pred) == len(ners)
                for label_idx, p, ner in zip(pred, prob, ners):
                    ner['attr'] = self.idx2label[label_idx]
                    ner['prob'] = p.tolist()

        with codecs.open(target_file, 'w', encoding='utf8') as f:
            for d in self.ori_data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':

    test_file = '../data/testb.txt'
    vocab_file = '../data/vocab.txt'

    config = '../saves/lawformer/config.json'
    init_checkpoint = '../saves/lawformer/model_lawformer.bin'

    target_file = '../data/testb_lawformer.txt'

    predictor = Predictor(test_file, vocab_file, config, init_checkpoint)
    predictor.pred(target_file)
