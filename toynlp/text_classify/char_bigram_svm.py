from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from toynlp.chinese_utils import char_word_tokenize


class CharBigramSVMClassifier:

    def __init__(self, max_features=5000):
        self.segment = lambda X: [' '.join(char_word_tokenize(text)) for text in X]
        self.pipe = Pipeline([('count', CountVectorizer(ngram_range=(2, 2),
                                                        token_pattern='\\b\\w+\\b')),
                              ('chi', SelectKBest(chi2, max_features)),
                              ('tfidf', TfidfTransformer())])
        self.clf = SVC(C=10)

    def fit(self, X_train, y_train):
        X_train = self.segment(X_train)
        X_train = self.pipe.fit_transform(X_train, y_train)
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        X = self.segment(X)
        X = self.pipe.transform(X)
        return self.clf.predict(X)
