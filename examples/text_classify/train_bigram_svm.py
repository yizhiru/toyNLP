# coding=gbk
import os

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from toynlp import utils
from toynlp.text_classify import CharBigramSVMClassifier


def load_data(path):
    lines = utils.read_lines(path)
    X = []
    y = []
    for idx, line in enumerate(lines):
        parts = line.strip().split('\t', 1)
        if len(parts) < 2:
            print(idx, parts, line)
        X.append(parts[1])
        y.append(parts[0])
    return X, y


root_path = '../../data/classify/small'
X_train, y_train = load_data(os.path.join(root_path, 'train.txt'))
X_test, y_test = load_data(os.path.join(root_path, 'test.txt'))

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

clf = CharBigramSVMClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
