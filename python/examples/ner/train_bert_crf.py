import argparse
import os
import sys

import keras
from seqeval.metrics import classification_report

_PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, _PYTHON_DIR)
_PROJECT_DIR = os.path.join(_PYTHON_DIR, '..')

from toynlp.ner import BertCRF  # noqa: E402
from toynlp import helper  # noqa: E402
from toynlp import utils  # noqa: E402

NER_DATA_DIR = os.path.join(_PROJECT_DIR, 'data', 'ner')


def parse_args():
    parser = argparse.ArgumentParser(description='BERT-CRF NER 训练与评估')
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--bert-output-layer-num', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device-map', type=str, default='3')
    parser.add_argument('--bert-model-path', type=str, default='chinese_L-12_H-768_A-12')
    parser.add_argument('--output-path', type=str, default='ner_model')
    return parser.parse_args()


def main():
    args = parse_args()
    print(f'\n{"ARG":>20s}   VALUE\n{"_" * 50}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:>20s} = {v}')
    print()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    X_train, y_train = helper.load_sequence_pair_data(os.path.join(NER_DATA_DIR, 'train.txt'))
    X_test, y_test = helper.load_sequence_pair_data(os.path.join(NER_DATA_DIR, 'test.txt'))
    X_val, y_val = helper.load_sequence_pair_data(os.path.join(NER_DATA_DIR, 'dev.txt'))

    utils.remake_dir(args.output_path)

    label2idx = helper.parse_label_seqs_to_dict(y_train)
    model = BertCRF(
        args.bert_model_path,
        label2idx,
        sequence_len=args.seq_len,
        bert_output_layer_num=args.bert_output_layer_num,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_crf_accuracy', patience=8),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_path, 'ner.h5'),
            monitor='val_crf_accuracy',
            save_best_only=True,
            save_weights_only=False,
        ),
    ]
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fit_kwargs={'callbacks': callbacks},
    )
    model.save_dict(args.output_path)

    y_pred = model.predict(X_test, batch_size=args.batch_size)
    y_test = [sub[:min(args.seq_len - 2, len(sub))] for sub in y_test]
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()
