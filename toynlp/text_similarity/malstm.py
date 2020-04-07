import random
from typing import Dict, List

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

    def __build_model(self):
        def exponent_neg_manhattan_distance(x):
            """"""
            return K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))

        input_layer1 = layers.Input(shape=(None,), dtype='int32', name='input1')
        input_layer2 = layers.Input(shape=(None,), dtype='int32', name='input2')

        embedding_layer = layers.Embedding(input_dim=len(self.token2idx) + 1,
                                           output_dim=self.embedding_matrix.shape[1],
                                           mask_zero=True,
                                           weights=[self.embedding_matrix],
                                           trainable=self.embedding_matrix is None,
                                           name='embedding')
        lstm_layer = layers.LSTM(units=self.lstm_units,
                                 dropout=self.lstm_dropout_rate,
                                 recurrent_dropout=self.lstm_dropout_rate,
                                 return_sequences=False,
                                 name='lstm')
        x1 = lstm_layer(embedding_layer(input_layer1))
        x2 = lstm_layer(embedding_layer(input_layer2))
        output = keras.layers.Lambda(function=exponent_neg_manhattan_distance,
                                     output_shape=(1,),
                                     name='output')([x1, x2])

        model = keras.models.Model(inputs=[input_layer1, input_layer2],
                                   outputs=[output])
        model.compile(optimizer=keras.optimizers.Adam(self.lr),
                      loss=keras.losses.mean_squared_error,
                      metrics=['accuracy'])
        model.summary()
        return model

    def __tokenize(self, sentences: List[List[str]]) -> List[List[int]]:
        """词映射index"""

        def tokenize_sentence(sentence: List[str]) -> List[int]:
            tokens = [self.token2idx.get(token, self.token2idx[H.UNK]) for token in sentence]
            return tokens[:self.sequence_len]

        return [tokenize_sentence(sen) for sen in sentences]

    def __data_generator(self,
                         x: (List[List[str]], List[List[str]]),
                         y: List[float],
                         batch_size: int):
        while True:
            page_list = list(range((len(x) // batch_size) + 1))
            random.shuffle(page_list)
            for page in page_list:
                start_index = page * batch_size
                end_index = start_index + batch_size
                target_x1 = x[0][start_index: end_index]
                target_x2 = x[1][start_index: end_index]
                target_y = y[start_index: end_index]
                if len(target_x1) == 0:
                    target_x = x[0: batch_size]
                    target_y = y[0: batch_size]

                tokenized_x1 = self.__tokenize(target_x[:, 0])
                padded_x1 = sequence.pad_sequences(tokenized_x1,
                                                   maxlen=self.sequence_len,
                                                   padding='post',
                                                   truncating='post',
                                                   value=self.token2idx[H.PAD])

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
