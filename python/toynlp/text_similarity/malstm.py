import math
import random
from typing import Dict, List, Tuple

import keras
import numpy as np
from keras import backend as K
from keras import layers
from keras_preprocessing import sequence

import toynlp.helper as H


class MaLSTM:

    def __init__(self,
                 token2idx: Dict,
                 embedding_matrix=None,
                 sequence_len=128,
                 lstm_units=128,
                 lstm_dropout_rate=0.4,
                 lr=0.001):
        self.token2idx = token2idx
        self.embedding_matrix = embedding_matrix
        self.sequence_len = sequence_len
        self.lstm_units = lstm_units
        self.lstm_dropout_rate = lstm_dropout_rate
        self.lr = lr
        self.model: keras.models.Model = None

    def _build_model(self):
        def exponent_neg_manhattan_distance(vects):
            return K.exp(-K.sum(K.abs(vects[0] - vects[1]), axis=1, keepdims=True))

        input_layer1 = layers.Input(shape=(None,), dtype='int32', name='input1')
        input_layer2 = layers.Input(shape=(None,), dtype='int32', name='input2')

        embedding_layer = layers.Embedding(
            input_dim=len(self.token2idx) + 1,
            output_dim=self.embedding_matrix.shape[1],
            mask_zero=True,
            weights=[self.embedding_matrix],
            trainable=self.embedding_matrix is None,
            name='embedding',
        )
        lstm_layer = layers.LSTM(
            units=self.lstm_units,
            dropout=self.lstm_dropout_rate,
            recurrent_dropout=self.lstm_dropout_rate,
            return_sequences=False,
            name='lstm',
        )
        x1 = lstm_layer(embedding_layer(input_layer1))
        x2 = lstm_layer(embedding_layer(input_layer2))
        output = keras.layers.Lambda(
            function=exponent_neg_manhattan_distance,
            output_shape=(1,),
            name='output',
        )([x1, x2])

        model = keras.models.Model(inputs=[input_layer1, input_layer2], outputs=[output])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=keras.losses.mean_squared_error,
            metrics=['accuracy'],
        )
        model.summary()
        self.model = model

    def _tokenize(self, sentences: List[List[str]]) -> List[List[int]]:
        """词映射index"""
        def tokenize_sentence(sentence: List[str]) -> List[int]:
            tokens = [self.token2idx.get(token, self.token2idx[H.UNK]) for token in sentence]
            return tokens[:self.sequence_len]

        return [tokenize_sentence(sen) for sen in sentences]

    def _pad(self, token_seqs: List[List[int]]) -> np.ndarray:
        return sequence.pad_sequences(
            token_seqs, maxlen=self.sequence_len,
            padding='post', truncating='post',
            value=self.token2idx[H.PAD],
        )

    def _data_generator(self,
                        x: Tuple[List[List[str]], List[List[str]]],
                        y: List[float],
                        batch_size: int):
        """x = (sentences_left, sentences_right), y = similarity scores"""
        n = len(x[0])
        while True:
            indices = list(range(n))
            random.shuffle(indices)
            for start in range(0, n, batch_size):
                batch_idx = indices[start: start + batch_size]
                if not batch_idx:
                    continue
                batch_x1 = [x[0][i] for i in batch_idx]
                batch_x2 = [x[1][i] for i in batch_idx]
                batch_y = [y[i] for i in batch_idx]

                padded_x1 = self._pad(self._tokenize(batch_x1))
                padded_x2 = self._pad(self._tokenize(batch_x2))

                yield ([padded_x1, padded_x2], np.array(batch_y))

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            batch_size=64,
            epochs=10,
            fit_kwargs: Dict = None):
        if self.model is None:
            self._build_model()

        n_train = len(X_train[0])
        if n_train < batch_size:
            batch_size = n_train // 2

        train_generator = self._data_generator(X_train, y_train, batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if X_val:
            val_generator = self._data_generator(X_val, y_val, batch_size)
            fit_kwargs['validation_data'] = val_generator
            fit_kwargs['validation_steps'] = math.ceil(len(X_val[0]) / batch_size)

        self.model.fit(
            train_generator,
            steps_per_epoch=math.ceil(n_train / batch_size),
            epochs=epochs,
            **fit_kwargs,
        )

    def predict(self,
                sentences_left: List[List[str]],
                sentences_right: List[List[str]],
                batch_size=64) -> np.ndarray:
        """预测两组句子的相似度"""
        padded_left = self._pad(self._tokenize(sentences_left))
        padded_right = self._pad(self._tokenize(sentences_right))
        return self.model.predict([padded_left, padded_right], batch_size=batch_size)
