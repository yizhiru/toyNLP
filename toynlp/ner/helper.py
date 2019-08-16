import os
from collections import defaultdict
import codecs

PAD = "[PAD]"
UNK = "[UNK]"

OTHER_LABEL = 'O'


def load_sequence_pair_data(file_path, sep=' '):
    char_seqs, label_seqs = [[]], [[]]
    with codecs.open(file_path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            # empty string
            if not line:
                if not char_seqs or len(char_seqs[-1]) > 0:
                    char_seqs.append([])
                    label_seqs.append([])
                continue
            parts = line.split(sep)
            if parts[0] != '-DOCSTART-':
                char_seqs[-1].append(parts[0])
                label_seqs[-1].append(parts[1])
        if not char_seqs[-1]:
            char_seqs.pop()
            label_seqs.pop()
    return char_seqs, label_seqs


def parse_label_seqs_to_dict(label_seqs):
    labels = set()
    for label_seq in label_seqs:
        for label in label_seq:
            labels.add(label)
    label2idx = {PAD: 0}
    for l in labels:
        label2idx[l] = len(label2idx)
    return label2idx


def parse_char_seqs_to_dict(char_seqs, min_freq=1):
    char2freq_dict = defaultdict(int)
    for seq in char_seqs:
        for ch in seq:
            char2freq_dict[ch] += 1
    token2idx = {PAD: 0, UNK: 1}
    for ch, freq in char2freq_dict.items():
        if freq >= min_freq:
            token2idx[ch] = len(token2idx)
    return token2idx


def read_bert_vocab(bert_model_path):
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token2idx = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        tokens = f.read().splitlines()
    for word in tokens:
        token2idx[word] = len(token2idx)
    return token2idx
