import numpy as np
from copy import deepcopy as dp
import re
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['sents_tokens'])

    def __getitem__(self, index):
        return {
            "sents_tokens": self.tokenizer.convert_tokens_to_ids(self.data['sents_tokens'][index]),
            "sents_other_feats": self.data["sents_other_feats"][index],
            "sents_len": self.data['sents_len'][index],
            "sents_tokens_start": self.data['sents_tokens_start'][index],
            "sents_label": self.data['sents_label'][index]
        }

    def collate_fn(self, data):
        data_padded = {}
        batch_max_len = max([len(data_item['sents_tokens']) for data_item in data])+1
        for k in data[0].keys():
            data_padded[k] = []
        for data_item in data:
            tokens_padded = np.zeros(batch_max_len, dtype=np.int64)
            other_feats_padded = np.zeros((batch_max_len,3), dtype=np.float)
            lable_padded = np.zeros(batch_max_len, dtype=np.float)
            tokens_start_padded = dp(tokens_padded)
            for i in range(len(data_item['sents_tokens'])):
                tokens_padded[i] = data_item['sents_tokens'][i]
                other_feats_padded[i]= data_item['sents_other_feats'][i]
            for i in range(len(data_item['sents_tokens_start'])):
                tokens_start_padded[i] = data_item['sents_tokens_start'][i]
                lable_padded[i] = data_item['sents_label'][i]
                """
                if i == (len(data_item['sents_tokens_start']) - 1):
                    lable_padded[(data_item['sents_tokens_start'][i] - 1)] = data_item['sents_label'][i]
                    break
                lable_padded[(data_item['sents_tokens_start'][i] - 1):(data_item['sents_tokens_start'][i + 1] - 1)] = data_item['sents_label'][i]
                """
            data_padded["sents_other_feats"].append(other_feats_padded)
            data_padded['sents_tokens'].append(tokens_padded)
            data_padded['sents_len'].append(data_item['sents_len'])
            data_padded['sents_tokens_start'].append(tokens_start_padded)
            data_padded['sents_label'].append(lable_padded)

        for k in data_padded.keys():
            data_padded[k] = torch.tensor(data_padded[k])
        return data_padded

class TestData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['sents_tokens'])

    def __getitem__(self, index):
        return {
            "sents_tokens": self.tokenizer.convert_tokens_to_ids(self.data['sents_tokens'][index]),
            "sents_other_feats": self.data["sents_other_feats"][index],
            "sents_len": self.data['sents_len'][index],
            "sents_tokens_start": self.data['sents_tokens_start'][index],
        }

    def collate_fn(self, data):
        data_padded = {}
        batch_max_len = max([len(data_item['sents_tokens']) for data_item in data])+1
        for k in data[0].keys():
            data_padded[k] = []
        for data_item in data:
            tokens_padded = np.zeros(batch_max_len, dtype=np.int64)
            other_feats_padded = np.zeros((batch_max_len,3), dtype=np.float)
            tokens_start_padded = dp(tokens_padded)
            for i in range(len(data_item['sents_tokens'])):
                tokens_padded[i] = data_item['sents_tokens'][i]
                other_feats_padded[i]= data_item['sents_other_feats'][i]
            for i in range(len(data_item['sents_tokens_start'])):
                tokens_start_padded[i] = data_item['sents_tokens_start'][i]

            data_padded["sents_other_feats"].append(other_feats_padded)
            data_padded['sents_tokens'].append(tokens_padded)
            data_padded['sents_len'].append(data_item['sents_len'])
            data_padded['sents_tokens_start'].append(tokens_start_padded)

        for k in data_padded.keys():
            data_padded[k] = torch.tensor(data_padded[k])
        return data_padded

def word2features(word):
    features=[]
    if word[0].isupper() and not word.isupper():
        features.append(0)
    elif not word[0].isupper():
        features.append(1)
    elif word.isupper():
        features.append(2)

    if re.search('\w',word) is None:
        features.append(0)
    elif re.search('\W',word) is None:
        features.append(1)
    else:
        features.append(2)
    return features

def read_data(filename, tokenizer):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('datasets/data_pos/pos.txt', 'r', encoding='utf-8') as f:
        pos_list = [line.strip() for line in f.readlines()]
    pre_lno = 0
    sents_len = []
    sent_other_feats, sents_other_feats=[], []
    sent_label, sents_label = [], []
    sent_tokens, sents_tokens = [], []
    sent_tokens_start, sents_tokens_start = [], []
    sent_tokens.append('<s>')
    sent_other_feats.append(word2features('<s>')+[pos_list.index('UNK')])

    sent_words_id, sent_words = [], []
    sents_words_id, sents_words = [], []

    for line in lines:
        if not line.isspace():
            id, word, bio, freq, prob, pos = line.strip().split()
            prob = float(prob)
            _, sid, lid, wid = id.split('-')
            if int(lid) != pre_lno:
                pre_lno = int(lid)
                sent_tokens.append('</s>')
                sent_other_feats.append(word2features('</s>')+[pos_list.index('UNK')])

                sents_words_id.append(sent_words_id)
                sents_words.append(sent_words)
                sents_label.append(sent_label)
                sents_other_feats.append(sent_other_feats)
                sents_len.append(len(sent_tokens))
                sents_tokens.append(sent_tokens)
                sents_tokens_start.append(sent_tokens_start)

                sent_tokens = ['<s>']
                sent_label = []
                sent_tokens_start = []
                sent_other_feats = []
                sent_other_feats.append(word2features('<s>')+[pos_list.index('UNK')])
                sent_words_id, sent_words = [], []

            sent_words.append(word)
            sent_words_id.append(id)
            sent_label.append(prob)
            sent_tokens_start.append(len(sent_tokens))
            tokens=tokenizer.tokenize(word)
            sent_other_feats.extend([word2features(word)+[int(pos)]]*len(tokens))
            sent_tokens.extend(tokens)
        else:
            pre_lno=0
            sent_tokens.append('</s>')
            sent_other_feats.append(word2features('</s>') + [pos_list.index('UNK')])

            sents_words_id.append(sent_words_id)
            sents_words.append(sent_words)
            sents_label.append(sent_label)
            sents_other_feats.append(sent_other_feats)
            sents_len.append(len(sent_tokens))
            sents_tokens.append(sent_tokens)
            sents_tokens_start.append(sent_tokens_start)

            sent_tokens = ['<s>']
            sent_label = []
            sent_tokens_start = []
            sent_other_feats = []
            sent_other_feats.append(word2features('<s>') + [pos_list.index('UNK')])
            sent_words_id, sent_words = [], []


    return {
        "sents_tokens": sents_tokens,
        "sents_tokens_start": sents_tokens_start,
        "sents_label": sents_label,
        'sents_other_feats': sents_other_feats,
        "sents_len": sents_len
    },sents_words_id, sents_words

def read_test_data(filename, tokenizer):
    with open('datasets/data_pos/pos.txt', 'r', encoding='utf-8') as f:
        pos_list = [line.strip() for line in f.readlines()]

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pre_lno = 0
    sents_len = []
    sent_other_feats, sents_other_feats=[], []
    sent_tokens, sents_tokens = [], []
    sent_tokens_start, sents_tokens_start = [], []
    sent_tokens.append('<s>')
    sent_other_feats.append(word2features('<s>')+[pos_list.index('UNK')])

    sent_words_id, sent_words = [], []
    sents_words_id, sents_words = [], []

    for line in lines:
        if not line.isspace():
            id, word, pos= line.strip().split()
            _, sid, lid, wid = id.split('-')
            if int(lid) != pre_lno:
                pre_lno = int(lid)
                sent_tokens.append('</s>')
                sent_other_feats.append(word2features('</s>')+[pos_list.index('UNK')])

                sents_words_id.append(sent_words_id)
                sents_words.append(sent_words)
                sents_other_feats.append(sent_other_feats)
                sents_len.append(len(sent_tokens))
                sents_tokens.append(sent_tokens)
                sents_tokens_start.append(sent_tokens_start)

                sent_tokens ,sent_tokens_start, sent_other_feats= ['<s>'], [], [word2features('<s>')+[pos_list.index('UNK')]]
                sent_words_id, sent_words = [], []

            sent_words.append(word)
            sent_words_id.append(id)
            sent_tokens_start.append(len(sent_tokens))
            tokens=tokenizer.tokenize(word)
            sent_other_feats.extend([word2features(word)+[int(pos)]]*len(tokens))
            sent_tokens.extend(tokens)
        else:
            pre_lno=0
            sent_tokens.append('</s>')
            sent_other_feats.append(word2features('</s>') + [pos_list.index('UNK')])

            sents_words_id.append(sent_words_id)
            sents_words.append(sent_words)
            sents_other_feats.append(sent_other_feats)
            sents_len.append(len(sent_tokens))
            sents_tokens.append(sent_tokens)
            sents_tokens_start.append(sent_tokens_start)

            sent_tokens = ['<s>']
            sent_tokens_start = []
            sent_other_feats = []
            sent_other_feats.append(word2features('<s>') + [pos_list.index('UNK')])
            sent_words_id, sent_words = [], []

    return {
        "sents_tokens": sents_tokens,
        "sents_tokens_start": sents_tokens_start,
        'sents_other_feats': sents_other_feats,
        "sents_len": sents_len
    }, sents_words_id, sents_words
