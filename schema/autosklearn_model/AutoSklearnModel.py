from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
import autosklearn.classification
from autosklearn.metrics import balanced_accuracy
import time


class AutoSklearnModel(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self, resampling_strategy='holdout', resampling_strategy_arguments=None):
        self.autosklearn_model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=20*60, #20*60
                                                                             n_jobs=20,#20
                                                                             ml_memory_limit=200072,
                                                                             ensemble_memory_limit=200072,
                                                                             initial_configurations_via_metalearning=0,
                                                                             resampling_strategy=resampling_strategy,
                                                                             resampling_strategy_arguments=resampling_strategy_arguments,
                                                                             exclude_preprocessors=['fast_ica'],
                                                                             tmp_folder='/home/neutatz/data/clean_auto/tmp/' + 'tmp' + str(time.time()),
                                                                             output_folder='/home/neutatz/data/clean_auto/out/' + 'out' + str(time.time()),
                                                                             #delete_tmp_folder_after_terminate = False,
                                                                             #delete_output_folder_after_terminate = False,
                                                                             )



    def fit(self, X, y, feat_type=None):
        self.autosklearn_model.fit(X.copy(), y.copy(), metric=balanced_accuracy, feat_type=feat_type)
        self.autosklearn_model.refit(X.copy(), y.copy())

    def predict(self, X):
        return self.autosklearn_model.predict(X)
