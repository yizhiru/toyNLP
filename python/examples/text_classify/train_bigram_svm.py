import argparse
import os
import sys

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

_PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, _PYTHON_DIR)
_PROJECT_DIR = os.path.join(_PYTHON_DIR, '..')

from toynlp import utils  # noqa: E402
from toynlp.text_classify import CharBigramSVMClassifier  # noqa: E402


def load_data(path):
    lines = utils.read_lines(path)
    X, y = [], []
    for idx, line in enumerate(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            print(f'[WARN] line {idx} malformed: {parts}')
            continue
        X.append(parts[1])
        y.append(parts[0])
    return X, y


def parse_args():
    default_data_dir = os.path.join(_PROJECT_DIR, 'data', 'classify', 'small')
    parser = argparse.ArgumentParser(description='char Bigram + SVM 文本分类训练与评估')
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                        help='训练/测试数据所在目录')
    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train = load_data(os.path.join(args.data_dir, 'train.txt'))
    X_test, y_test = load_data(os.path.join(args.data_dir, 'test.txt'))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    clf = CharBigramSVMClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
