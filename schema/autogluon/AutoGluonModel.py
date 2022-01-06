from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


class AutoGluonModel(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    def __init__(self):
        self.autogluon_model = TabularPredictor(label='my_target_label1234', eval_metric='balanced_accuracy')

    def fit(self, X, y):
        df = pd.DataFrame(data=X)
        label = 'my_target_label1234'
        df[label] = y
        my_data_train = TabularDataset(data=df)

        self.autogluon_model.fit(train_data=my_data_train, time_limit=10*60, presets='best_quality')


    def predict(self, X):
        df_test = pd.DataFrame(data=X)
        my_data_test = TabularDataset(data=df_test)
        return self.autogluon_model.predict(my_data_test)
