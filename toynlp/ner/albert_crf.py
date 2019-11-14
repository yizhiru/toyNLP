from typing import Dict

import keras
import keras_albert_model
from keras import backend as K
from keras.layers import TimeDistributed, Dense
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_losses
from keras_contrib.metrics import crf_accuracies

from toynlp import helper as H
from toynlp.ner import BertCRF


class AlbertCRF(BertCRF):

    def __init__(self,
                 albert_model_path=None,
                 label2idx: Dict = None,
                 sequence_len=128,
                 dense_units=256,
                 lr=0.001):
        super().__init__(albert_model_path,
                         label2idx,
                         sequence_len=sequence_len,
                         dense_units=dense_units,
                         lr=lr)

    def _build_model(self):
        self.token2idx = H.read_bert_vocab(self.bert_model_path)
        albert_model = keras_albert_model.load_brightmart_albert_zh_checkpoint(self.bert_model_path,
                                                                               seq_len=self.sequence_len,
                                                                               training=False)
        dense_layer = TimeDistributed(Dense(self.dense_units, activation=K.tanh),
                                      name='td_dense')(albert_model.output)
        crf_layer = CRF(units=len(self.label2idx), sparse_target=False, name='CRF')(dense_layer)
        model = keras.models.Model(inputs=albert_model.inputs, outputs=crf_layer)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=crf_losses.crf_loss,
                      metrics=[crf_accuracies.crf_accuracy])
        model.summary()
        self.model = model

    @classmethod
    def get_custom_objects(cls):
        custom_objects = keras_albert_model.get_custom_objects()
        custom_objects['CRF'] = CRF
        custom_objects['crf_loss'] = crf_losses.crf_loss
        custom_objects['crf_accuracy'] = crf_accuracies.crf_accuracy
        return custom_objects
