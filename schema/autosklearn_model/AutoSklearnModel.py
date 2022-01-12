from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
import autosklearn.classification
from autosklearn.metrics import balanced_accuracy


class AutoSklearnModel(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self):
        self.autosklearn_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=10*60,
                                                                             metric=balanced_accuracy,
                                                                             n_jobs=5,
                                                                             memory_limit=200072)

    def fit(self, X, y, feat_type=None):
        self.autosklearn_model.fit(X.copy(), y.copy())
        #self.autosklearn_model.refit(X.copy(), y.copy())

    def predict(self, X):
        return self.autosklearn_model.predict(X)
