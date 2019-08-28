import json
import os
import random
from typing import Dict, List

import keras
import keras_bert
import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_losses
from keras_contrib.metrics import crf_accuracies
from keras_preprocessing import sequence
from keras_self_attention import SeqSelfAttention

import toynlp.helper as H


class BiLSTMCRFModel:

    def __init__(self,
                 token2idx: Dict = None,
                 label2idx: Dict = None,
                 sequence_len=128,
                 embedding_dim=100,
                 lstm_units=256,
                 dense_units=256,
                 lr=0.001):
        self.token2idx = token2idx
        self.label2idx = label2idx
        if label2idx:
            self.idx2label = {v: k for k, v in label2idx.items()}
        else:
            self.idx2label = None
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.lr = lr
        self.model: keras.models.Model = None

    def __build_model(self):
        input_layer = keras.layers.Input(shape=(None,))
        embedding_layer = keras.layers.Embedding(input_dim=len(self.token2idx),
                                                 output_dim=self.embedding_dim,
                                                 mask_zero=True,
                                                 trainable=True,
                                                 name='Embedding')(input_layer)
        bilstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=self.lstm_units,
                                                                    recurrent_dropout=0.4,
                                                                    return_sequences=True),
                                                  name='Bi-LSTM')(embedding_layer)
        dense_layer = keras.layers.TimeDistributed(Dense(units=self.dense_units,
                                                         activation=K.relu),
                                                   name='td_dense')(bilstm_layer)
        crf_layer = CRF(units=len(self.label2idx), sparse_target=True, name='CRF')(dense_layer)
        model = keras.models.Model(inputs=input_layer, outputs=crf_layer)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=crf_losses.crf_loss,
                      metrics=[crf_accuracies.crf_accuracy])
        model.summary()
        self.model = model

    def __tokenize(self, sentences: List[List[str]]) -> List[List[int]]:
        """字符映射index"""

        def tokenize_sentence(sentence: List[str]) -> List[int]:
            tokens = [self.token2idx.get(token, self.token2idx[H.UNK]) for token in sentence]
            return tokens[:self.sequence_len]

        return [tokenize_sentence(sen) for sen in sentences]

    def __convert_label_seqs_to_idx(self, label_seqs: List[List[str]]) -> List[List[int]]:
        """label映射index"""

        def convert_label_seq(label_seq: List[str]):
            idx_seq = [self.label2idx[l] for l in label_seq]
            return idx_seq[:self.sequence_len]

        return [convert_label_seq(seq) for seq in label_seqs]

    def __convert_idx_seqs_to_label(self,
                                    idx_seqs: List[List[int]],
                                    raw_len_seqs: List[int]) -> List[List[str]]:
        """index映射index"""

        def __convert_idx_seq_to_label(indices, raw_length):
            indices = indices[: min(raw_length, self.sequence_len)]
            return [self.idx2label[i] for i in indices]

        label_seqs = []
        for idx_seq, raw_len in zip(idx_seqs, raw_len_seqs):
            label_seqs.append(__convert_idx_seq_to_label(idx_seq, raw_len))
        return label_seqs

    def __data_generator(self,
                         x: List[List[str]],
                         y: List[List[str]],
                         batch_size: int):
        while True:
            steps = (len(x) + batch_size - 1) // batch_size
            # shuffle data
            xy = list(zip(x, y))
            random.shuffle(xy)
            x, y = zip(*xy)
            for i in range(steps):
                batch_x = x[i * batch_size: (i + 1) * batch_size]
                batch_y = y[i * batch_size: (i + 1) * batch_size]

                tokenized_x = self.__tokenize(batch_x)
                idx_y = self.__convert_label_seqs_to_idx(batch_y)

                padded_x = sequence.pad_sequences(tokenized_x,
                                                  maxlen=self.sequence_len,
                                                  padding='post',
                                                  truncating='post',
                                                  value=self.token2idx[H.PAD])
                padded_y = sequence.pad_sequences(idx_y,
                                                  maxlen=self.sequence_len,
                                                  padding='post',
                                                  truncating='post',
                                                  value=self.label2idx[H.PAD])
                padded_y = np.reshape(padded_y, padded_y.shape + (1,))
                yield (padded_x, padded_y)

    def fit(self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64,
            epochs=10,
            fit_kwargs: Dict = None):
        if self.model is None:
            self.__build_model()

        if len(X_train) < batch_size:
            batch_size = len(X_train) // 2

        train_generator = self.__data_generator(X_train, y_train, batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if X_val:
            val_generator = self.__data_generator(X_val, y_val, batch_size)
            fit_kwargs['validation_data'] = val_generator
            fit_kwargs['validation_steps'] = len(X_val) // batch_size

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=len(X_train) // batch_size,
                                 epochs=epochs,
                                 **fit_kwargs)

    def predict(self,
                sentences: List[List[str]],
                batch_size=64) -> List[List[str]]:
        tokens = self.__tokenize(sentences)
        raw_len_seqs = [len(sentence) for sentence in sentences]
        padded_tokens = sequence.pad_sequences(tokens,
                                               maxlen=self.sequence_len,
                                               padding='post',
                                               truncating='post',
                                               value=self.token2idx[H.PAD])
        pred_prob_seqs = self.model.predict(padded_tokens, batch_size=batch_size)
        idx_seqs = pred_prob_seqs.argmax(-1)

        return self.__convert_idx_seqs_to_label(idx_seqs, raw_len_seqs)

    @classmethod
    def get_custom_objects(cls):
        custom_objects = keras_bert.get_custom_objects()
        custom_objects['SeqSelfAttention'] = SeqSelfAttention
        custom_objects['CRF'] = CRF
        custom_objects['crf_loss'] = crf_losses.crf_loss
        custom_objects['crf_accuracy'] = crf_accuracies.crf_accuracy
        return custom_objects

    def save_dict(self, dict_path):
        with open(os.path.join(dict_path, 'vocab.json'), 'w', encoding='utf8') as fw:
            fw.write(json.dumps(self.token2idx, indent=2, ensure_ascii=False))
        with open(os.path.join(dict_path, 'labels.json'), 'w', encoding='utf8') as fw:
            fw.write(json.dumps(self.label2idx, indent=2, ensure_ascii=False))

    @classmethod
    def load_model(cls, model_path, dict_path, sequence_len):
        agent = cls(sequence_len=sequence_len)
        agent.model = keras.models.load_model(model_path, custom_objects=cls.get_custom_objects())
        agent.model.summary()
        with open(os.path.join(dict_path, 'vocab.json'), 'r', encoding='utf8') as fr:
            agent.token2idx = json.load(fr)
        with open(os.path.join(dict_path, 'labels.json'), 'r', encoding='utf8') as fr:
            agent.label2idx = json.load(fr)
        agent.idx2label = {v: k for k, v in agent.label2idx.items()}
        return agent
