from schema.autosklearn_model.AutoSklearnModel import AutoSklearnModel
from schema.autogluon.AutoGluonModel import AutoGluonModel
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import copy
from sklearn.inspection import permutation_importance
import pickle

def get_X_y(data, target_label, drop_labels=[]):
    data_y = data[target_label]
    data_X = data.drop(target_label, 1)
    for drop_label in drop_labels:
        data_X = data.drop(drop_label, 1)
    return data_y, data_X

def eval(data, target_label, fold_ids, drop_labels=[], feat_type=None, use_autosklearn=True, mislabels_percent=0.0, file_name=None):
    data_y, data_X = get_X_y(data, target_label, drop_labels)

    if use_autosklearn:
        if type(feat_type) != type(None):
            for ci in range(len(feat_type)):
                if feat_type[ci] == 'Categorical':
                    data_X[data_X.columns[ci]] = data_X[data_X.columns[ci]].astype('category')



    scores = []
    result_models = []
    feature_importances = []
    for train_index, test_index in fold_ids:
        model = None

        y_train = copy.deepcopy(data_y.values[train_index])
        if mislabels_percent > 0:
            # find indices for each class
            class_values = np.unique(y_train)
            mislabels_percent_per_class = mislabels_percent / len(class_values)

            indices_all = []
            for class_i in range(len(class_values)):
                indices_class_i = np.where(y_train == class_values[class_i])[0]
                np.random.shuffle(indices_class_i)
                indices_all.append(indices_class_i)

            for class_i in range(len(class_values)):
                for change_i in range(int(mislabels_percent_per_class * len(y_train))):
                    class_choice = copy.deepcopy(class_values)
                    class_choice = np.delete(class_choice, [class_i], None)
                    new_value = np.random.choice(class_choice)
                    y_train[indices_all[class_i][change_i]] = new_value

            #print('error fraction: ' + str(np.sum(y_train != data_y.values[train_index]) / float(len(y_train))))

        if use_autosklearn:
            model = AutoSklearnModel()
            model.fit(X=data_X.iloc[train_index, :], y=y_train, feat_type=feat_type)
        else:
            model = AutoGluonModel()
            model.fit(X=data_X.iloc[train_index, :], y=y_train)

        result_models.append(copy.deepcopy(model))

        y_pred = model.predict(data_X.iloc[test_index])
        y_true = data_y.values[test_index]
        scores.append(balanced_accuracy_score(y_true, y_pred))
        print(scores)

        #compute permutation importance

        r = permutation_importance(model, data_X.iloc[test_index], y_true, n_repeats=10, random_state=0)
        feature_importances.append(copy.deepcopy(r))

        if type(None) != type(file_name):
            result_dict = {}
            result_dict['scores'] = scores
            result_dict['models'] = result_models
            result_dict['feature_importances'] = feature_importances

            with open(file_name, "wb") as pickle_model_file:
                pickle.dump(result_dict, pickle_model_file)



    return scores

def get_fold_ids(data, target_label, drop_labels=[]):
    data_y, data_X = get_X_y(data, target_label, drop_labels)

    skf = StratifiedKFold(n_splits=5)
    fold_ids = list(skf.split(data_X, data_y))
    return fold_ids

def get_feat_type(data, target_label, drop_labels=[]):
    data_y, data_X = get_X_y(data, target_label, drop_labels)
    feat_type = [
        'Categorical' if str(x) == 'object' else 'Numerical'
        for x in data_X.dtypes
    ]
    return feat_type

def run(clean_path, dirty_path, target_label, drop_labels=[], use_autosklearn=True, mislabel_percent=0.0, file_name=None):
    holoclean_train = pd.read_csv(clean_path)
    dirty_train = None
    if mislabel_percent == 0.0:
        dirty_train = pd.read_csv(dirty_path)
    #assert len(holoclean_train) == len(dirty_train)

    fold_ids = get_fold_ids(holoclean_train, target_label, drop_labels)
    feat_type = get_feat_type(holoclean_train, target_label, drop_labels)
    if mislabel_percent > 0.0:
        dirty_scores = eval(holoclean_train, target_label, fold_ids, drop_labels, feat_type, use_autosklearn, mislabels_percent=mislabel_percent, file_name=file_name + '_dirty.p')
    else:
        dirty_scores = eval(dirty_train, target_label, fold_ids, drop_labels, feat_type, use_autosklearn, file_name=file_name + '_dirty.p')
    clean_scores = eval(holoclean_train, target_label, fold_ids, drop_labels, feat_type, use_autosklearn, file_name=file_name + '_clean.p')

    print('dirty scores: ' + str(dirty_scores))
    print('clean scores: ' + str(clean_scores))

    #print('number of errors: ' + str(np.sum(holoclean_train != dirty_train)))