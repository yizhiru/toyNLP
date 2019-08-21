import argparse
import os
import sys

import keras
from seqeval.metrics import classification_report

sys.path.append('../../../')

from toynlp.ner.model import BiLSTMCRFModel
from toynlp.ner import helper

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('-seq_len', type=int, default=128)
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-device_map', type=str, default='3')
parser.add_argument('-model_path', type=str, default='ner_model')
args = parser.parse_args()
param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

root_path = '../../../data/ner'
X_train, y_train = helper.load_sequence_pair_data(os.path.join(root_path, 'train.txt'))
X_test, y_test = helper.load_sequence_pair_data(os.path.join(root_path, 'test.txt'))
X_val, y_val = helper.load_sequence_pair_data(os.path.join(root_path, 'dev.txt'))

char2idx = helper.parse_char_seqs_to_dict(X_train + X_val)
label2idx = helper.parse_label_seqs_to_dict(y_train)
model = BiLSTMCRFModel(char2idx,
                       label2idx,
                       sequence_len=args.seq_len)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_crf_accuracy', patience=8),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_path, 'ner.h5'),
                                    monitor='val_crf_accuracy',
                                    save_best_only=True,
                                    save_weights_only=False)
]
model.fit(X_train,
          y_train,
          X_val=X_val,
          y_val=y_val,
          epochs=args.epochs,
          batch_size=args.batch_size,
          fit_kwargs={'callbacks': callbacks}
          )

# save word dict and label dict
model.save_dict(args.model_path)

# load model
model = BiLSTMCRFModel.load_model(os.path.join(args.model_path, 'ner.h5'),
                                  dict_path=args.model_path,
                                  sequence_len=args.seq_len)

y_pred = model.predict(X_test, batch_size=args.batch_size)
y_test = [sub[:min(args.seq_len, len(sub))] for sub in y_test]
print(classification_report(y_test, y_pred, digits=4))
