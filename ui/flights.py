from skimage.io import imread
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from glob import glob
from schema.autosklearn_model.AutoSklearnModel import AutoSklearnModel
import sklearn

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
import copy
from sklearn.inspection import permutation_importance
import pickle
from sklearn.model_selection import GroupKFold

def to_str(data):
    return str(data)


if __name__ == '__main__':

    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    clean = pd.read_csv('/home/neutatz/phd2/clean_autoMl/data/flights_clean.csv')
    dirty = pd.read_csv('/home/neutatz/phd2/clean_autoMl/data/flights_dirty.csv')

    #grouping
    cols = ['Airline', 'Flight Number', 'dep_airport', 'arr_airport']
    new_group = clean[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1).values

    print(new_group)

    print(np.sum(clean['more_than_5_minutes_delay']) / len(clean))
    print(np.sum(dirty['more_than_5_minutes_delay']) / len(dirty))

    y_clean = clean['more_than_5_minutes_delay'].values
    y_dirty = dirty['more_than_5_minutes_delay'].values

    clean = clean.drop(columns=['more_than_5_minutes_delay'])
    dirty = dirty.drop(columns=['more_than_5_minutes_delay'])

    assert len(clean) == len(dirty)

    skf = GroupKFold(n_splits=5)
    fold_ids = list(skf.split(clean.values, y_clean, groups=new_group))

    scores_clean = []
    scores_dirty = []

    y_val = preprocessing.LabelEncoder().fit_transform(y_clean)

    data_X = clean

    feat_type = [
        'Categorical' if str(x) == 'object' else 'Numerical'
        for x in data_X.dtypes
    ]

    for ci in range(len(feat_type)):
        if feat_type[ci] == 'Categorical':
            data_X[data_X.columns[ci]] = data_X[data_X.columns[ci]].apply(to_str)

    data_X_val = data_X.values

    for ci in range(len(feat_type)):
        if feat_type[ci] == 'Categorical':
            my_encoder = preprocessing.LabelEncoder()
            data_X_val[:, ci] = my_encoder.fit_transform(data_X_val[:, ci])
            for class_i in range(len(my_encoder.classes_)):
                if my_encoder.classes_[class_i] == 'nan':
                    data_X_val[data_X_val[:, ci] == class_i, ci] = np.NaN

    data_X_val = data_X_val.astype('float64')

    scores = []
    result_models = []
    feature_importances = []

    for train_index, test_index in fold_ids:
        #resampling_strategy = 'holdout'
        resampling_strategy_arguments = None

        print(data_X_val[train_index, :])

        skf_new = GroupKFold(n_splits=3)
        fold_ids_new = list(skf_new.split(data_X_val[train_index, :], y=y_val[train_index], groups=new_group[train_index]))


        print(len(fold_ids_new[0][1]))

        resampling_strategy = sklearn.model_selection.PredefinedSplit(
            test_fold=fold_ids[0][1]
        )

        model = AutoSklearnModel(resampling_strategy=resampling_strategy,
                                 resampling_strategy_arguments=resampling_strategy_arguments)
        model.fit(X=data_X_val[train_index, :], y=y_val[train_index], feat_type=feat_type)

        result_models.append(copy.deepcopy(model))

        y_pred = model.predict(data_X_val[test_index])
        y_true = y_clean[test_index]
        scores.append(balanced_accuracy_score(y_true, y_pred))
        print(scores)

        # compute permutation importance

        r = permutation_importance(model, data_X_val[test_index], y_true, n_repeats=10, random_state=0)
        feature_importances.append(copy.deepcopy(r))

        if type(None) != type(file_name):
            result_dict = {}
            result_dict['scores'] = scores
            result_dict['models'] = result_models
            result_dict['feature_importances'] = feature_importances

            with open(file_name, "wb") as pickle_model_file:
                pickle.dump(result_dict, pickle_model_file)
