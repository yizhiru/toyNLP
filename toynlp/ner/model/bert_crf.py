import json
import logging
import os
import random
from typing import Dict, List

import keras
import keras_bert
import numpy as np
from keras import backend as K
from keras.layers import TimeDistributed, Dense
from keras_bert import bert
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_losses
from keras_contrib.metrics import crf_accuracies
from keras_preprocessing import sequence

import toynlp.ner.helper as H


class BertCRFModel:

    def __init__(self,
                 bert_model_path=None,
                 label2idx: Dict = None,
                 sequence_len=128,
                 bert_output_layer_num=1,
                 lstm_units=256,
                 lr=0.001):
        self.bert_model_path = bert_model_path
        self.label2idx = label2idx
        if label2idx:
            self.idx2label = {v: k for k, v in label2idx.items()}
        else:
            self.idx2label = None
        self.sequence_len = sequence_len
        self.bert_output_layer_num = bert_output_layer_num
        self.lstm_units = lstm_units
        self.lr = lr
        self.model: keras.models.Model = None

    def __load_bert_model(self):
        config_path = os.path.join(self.bert_model_path, 'bert_config.json')
        check_point_path = os.path.join(self.bert_model_path, 'bert_model.ckpt')
        logging.info('loading bert model from {}\n'.format(self.bert_model_path))
        bert_model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                                   check_point_path,
                                                                   seq_len=self.sequence_len,
                                                                   output_layer_num=self.bert_output_layer_num,
                                                                   training=False,
                                                                   trainable=False)
        return bert_model

    def __build_model(self):
        bert_model = self.__load_bert_model()
        self.token2idx = H.read_bert_vocab(self.bert_model_path)
        dense_layer = TimeDistributed(Dense(128, activation=K.tanh, name='dense'))(bert_model.output)
        crf_layer = CRF(units=len(self.label2idx), sparse_target=False, name='CRF')(dense_layer)
        model = keras.models.Model(inputs=bert_model.inputs, outputs=crf_layer)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=crf_losses.crf_loss,
                      metrics=[crf_accuracies.crf_accuracy])
        model.summary()
        self.model = model

    def __tokenize(self, sentences: List[List[str]]) -> List[List[int]]:
        def tokenize_sentence(sentence: List[str]) -> List[int]:
            tokens = [self.token2idx.get(token, self.token2idx[bert.TOKEN_UNK]) for token in sentence]
            # truncate, CLS ... token .. SEP
            tokens = tokens[:self.sequence_len - 2]
            tokens = [self.token2idx[bert.TOKEN_CLS]] + tokens + [self.token2idx[bert.TOKEN_SEP]]
            return tokens

        return [tokenize_sentence(sen) for sen in sentences]

    def __convert_label_seqs_to_idx(self, label_seqs: List[List[str]]) -> List[List[int]]:
        def convert_label_seq(label_seq: List[str]):
            idx_seq = [self.label2idx[l] for l in label_seq]
            idx_seq = idx_seq[:self.sequence_len - 2]
            idx_seq = [self.label2idx[H.OTHER_LABEL]] + idx_seq + [self.label2idx[H.OTHER_LABEL]]
            return idx_seq

        return [convert_label_seq(seq) for seq in label_seqs]

    def __convert_idx_seqs_to_label(self,
                                    idx_seqs: List[List[int]],
                                    raw_len_seqs: List[int]) -> List[List[str]]:
        def __convert_idx_seq_to_label(indices, raw_length):
            indices = indices[1: min(raw_length, self.sequence_len - 2) + 1]
            return [self.idx2label[i] for i in indices]

        label_seqs = []
        for idx_seq, raw_len in zip(idx_seqs, raw_len_seqs):
            label_seqs.append(__convert_idx_seq_to_label(idx_seq, raw_len))
        return label_seqs

    def __data_generator(self,
                         x: List[List[str]],
                         y: List[List[str]],
                         batch_size: int = 64):
        while True:
            page_list = list(range((len(x) // batch_size) + 1))
            random.shuffle(page_list)
            for page in page_list:
                start_index = page * batch_size
                end_index = start_index + batch_size
                target_x = x[start_index: end_index]
                target_y = y[start_index: end_index]
                if len(target_x) == 0:
                    target_x = x[0: batch_size]
                    target_y = y[0: batch_size]

                tokenized_x = self.__tokenize(target_x)
                idx_y = self.__convert_label_seqs_to_idx(target_y)

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
                one_hot_y = keras.utils.to_categorical(padded_y, num_classes=len(self.label2idx))

                padded_x_right = np.zeros(shape=padded_x.shape)
                x_input_data = [padded_x, padded_x_right]
                yield (x_input_data, one_hot_y)

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

        x = [padded_tokens, np.zeros(shape=padded_tokens.shape)]

        pred_prob_seqs = self.model.predict(x, batch_size=batch_size)
        idx_seqs = pred_prob_seqs.argmax(-1)

        return self.__convert_idx_seqs_to_label(idx_seqs, raw_len_seqs)

    @classmethod
    def get_custom_objects(cls):
        custom_objects = keras_bert.get_custom_objects()
        custom_objects['CRF'] = CRF
        custom_objects['crf_loss'] = crf_losses.crf_loss
        custom_objects['crf_accuracy'] = crf_accuracies.crf_accuracy
        return custom_objects

    def save_dict(self, dict_path):
        with open(os.path.join(dict_path, 'labels.json'), 'w', encoding='utf8') as fw:
            fw.write(json.dumps(self.label2idx, indent=2, ensure_ascii=False))

    @classmethod
    def load_model(cls, model_path, bert_model_path, dict_path, sequence_len):
        agent = cls(sequence_len=sequence_len)
        agent.model = keras.models.load_model(model_path, custom_objects=cls.get_custom_objects())
        agent.model.summary()
        agent.token2idx = H.read_bert_vocab(bert_model_path)
        with open(os.path.join(dict_path, 'labels.json'), 'r', encoding='utf8') as fr:
            agent.label2idx = json.load(fr)
        agent.idx2label = {v: k for k, v in agent.label2idx.items()}
        return agent
