import argparse
import os
import sys

import keras
from seqeval.metrics import classification_report

from ner.model import BertCRFModel
from toynlp.ner import helper

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('-seq_len', type=int, default=100)
parser.add_argument('-bert_output_layer_num', type=int, default=4)
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-device_map', type=str, default='3')
parser.add_argument('-bert_model_path', type=str, default='chinese_L-12_H-768_A-12')
parser.add_argument('-model_path', type=str, default='ner_model')
args = parser.parse_args()
param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

root_path = '../../../data/ner'
X_train, y_train = helper.load_sequence_pair_data(os.path.join(root_path, 'train.txt'))
X_test, y_test = helper.load_sequence_pair_data(os.path.join(root_path, 'test.txt'))
X_val, y_val = helper.load_sequence_pair_data(os.path.join(root_path, 'dev.txt'))

label2idx = helper.parse_label_seqs_to_dict(y_train)
model = BertCRFModel(args.bert_model_path,
                     label2idx,
                     sequence_len=args.seq_len,
                     bert_output_layer_num=args.bert_output_layer_num)

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

# save labels
model.save_dict(args.model_path)

y_pred = model.predict(X_test, batch_size=args.batch_size)
y_test = [sub[:min(args.seq_len - 2, len(sub))] for sub in y_test]
print(classification_report(y_test, y_pred, digits=4))
