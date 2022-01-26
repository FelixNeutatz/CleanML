from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
import autosklearn.classification
from autosklearn.metrics import balanced_accuracy
import time


class AutoSklearnModel:
    def __init__(self, resampling_strategy='holdout', resampling_strategy_arguments=None):
        self.autosklearn_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=20*60,
                                                                             metric=balanced_accuracy,
                                                                             n_jobs=20,
                                                                             memory_limit=200072,
                                                                             resampling_strategy=resampling_strategy,
                                                                             exclude={'feature_preprocessor':["fast_ica"]},
                                                                             tmp_folder='/home/neutatz/data/clean_auto/tmp/' + 'tmp' + str(time.time()),
                                                                             output_folder='/home/neutatz/data/clean_auto/out/' + 'out' + str(time.time()),
                                                                             resampling_strategy_arguments=resampling_strategy_arguments)

    def fit(self, X, y, feat_type=None):
        self.autosklearn_model.fit(X.copy(), y.copy(), feat_type=feat_type)
        self.autosklearn_model.refit(X.copy(), y.copy())

    def predict(self, X):
        return self.autosklearn_model.predict(X)
