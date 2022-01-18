import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import glob
print(glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"))

pointer_dict = {}
pointer_dict['credit_outliers'] = {}
pointer_dict['credit_outliers']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Credit/outliers/dirty_train.csv'
pointer_dict['credit_outliers']['clean_path'] = '/home/neutatz/Software/CleanML/data/Credit/outliers/clean_HC_impute_holoclean_train.csv'
pointer_dict['credit_outliers']['target'] = "SeriousDlqin2yrs"
pointer_dict['credit_outliers']['drop_labels'] = []

pointer_dict['airbnb_outliers'] = {}
pointer_dict['airbnb_outliers']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/outliers/dirty_train.csv'
pointer_dict['airbnb_outliers']['clean_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/outliers/clean_HC_impute_holoclean_train.csv'
pointer_dict['airbnb_outliers']['target'] = 'Rating'
pointer_dict['airbnb_outliers']['drop_labels'] = []

pointer_dict['eeg_outliers'] = {}
pointer_dict['eeg_outliers']['dirty_path'] = '/home/neutatz/Software/CleanML/data/EEG/outliers/dirty_train.csv'
pointer_dict['eeg_outliers']['clean_path'] = '/home/neutatz/Software/CleanML/data/EEG/outliers/clean_HC_impute_holoclean_train.csv'
pointer_dict['eeg_outliers']['target'] = 'Eye'
pointer_dict['eeg_outliers']['drop_labels'] = []

pointer_dict['sensor_outliers'] = {}
pointer_dict['sensor_outliers']['dirty_path'] =  '/home/neutatz/Software/CleanML/data/Sensor/outliers/dirty_train.csv'
pointer_dict['sensor_outliers']['clean_path'] = '/home/neutatz/Software/CleanML/data/Sensor/outliers/clean_HC_impute_holoclean_train.csv'
pointer_dict['sensor_outliers']['target'] = "moteid"
pointer_dict['sensor_outliers']['drop_labels'] = []



my_latex_table = ''

def get_X_y(data, target_label, drop_labels=[]):
    data_X = data.drop(target_label, 1)
    for drop_label in drop_labels:
        data_X = data.drop(drop_label, 1)
    return data_X

def get_names(clean_path, drop_variables, target):
    holoclean_train = pd.read_csv(clean_path)
    #print(holoclean_train)
    df = get_X_y(holoclean_train, target, drop_variables)
    return df.columns


def get_errors(clean_path, dirty_path, target_label, drop_labels=[]):
    holoclean_train = pd.read_csv(clean_path)
    dirty_train = pd.read_csv(dirty_path)
    X_clean = get_X_y(holoclean_train, target_label, drop_labels)
    X_dirty = get_X_y(dirty_train, target_label, drop_labels)
    return np.sum(X_clean != X_dirty).values

def get_feature_importances(mypath):
    results = pickle.load(open(mypath, 'rb'))

    feature_importances = []
    for i in range(len(results['feature_importances'])):
        feature_importances.append(results['feature_importances'][i]['importances_mean'])
    #print(feature_importances)

    return np.average(feature_importances, axis=0)

for clean_file in glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"):
    try:
        task_name = clean_file.split('/')[-1][:-11]
        dirty_file = clean_file[:-7] + 'dirty.p'
        print(task_name)

        target = pointer_dict[task_name]['target']
        drop_labels = pointer_dict[task_name]['drop_labels']
        clean_path = pointer_dict[task_name]['clean_path']
        dirty_path = pointer_dict[task_name]['dirty_path']

        #try:

        feature_importances_average_clean = get_feature_importances(clean_file)
        feature_importances_average_dirty = get_feature_importances(dirty_file)

        fnames = get_names(clean_path, drop_labels, target)

        #print(feature_importances_average_clean)
        #print(feature_importances_average_dirty)


        feature_importances_average_clean_scaled = MinMaxScaler().fit_transform(feature_importances_average_clean.reshape(-1, 1)).flatten()
        feature_importances_average_dirty_scaled = MinMaxScaler().fit_transform(feature_importances_average_dirty.reshape(-1, 1)).flatten()
        #print('clean scaled: ' + str(feature_importances_average_clean_scaled))
        #print_list(feature_importances_average_clean_scaled)
        #print('dirty scaled: ' + str(feature_importances_average_dirty_scaled))
        #print_list(feature_importances_average_dirty_scaled)


        errors = get_errors(clean_path, dirty_path, target, drop_labels)
        #print(errors)
        #print('diff: ' + str(feature_importances_average_clean - feature_importances_average_dirty))
        #print('names: ' + str(fnames))

        sorted_ids = np.argsort(feature_importances_average_clean_scaled *-1)

        my_latex_table += '\\begin{table*}\n\\smallestfont\n\\centering\n\\caption{Impact on Feature Importance for dirty Data (%s).}\n\label{tab:featureimportance%s}\n' % (task_name.replace('_', ' '), task_name.replace('_', ''))
        my_latex_table += '\\begin{tabular}{@{}lccc@{}}\n'
        my_latex_table += '\\toprule\n'
        my_latex_table += 'Feature Names & Scaled Importance Clean & Scaled Importance Dirty & Number Errors \\\\'
        my_latex_table += '\\midrule\n'
        for my_index in range(len(fnames)):
            ii = sorted_ids[my_index]
            my_latex_table += '%s & %.2f & %.2f & %d \\\\ \n' % (fnames[ii], feature_importances_average_clean_scaled[ii], feature_importances_average_dirty_scaled[ii], errors[ii])
        my_latex_table += '\\bottomrule\n'
        my_latex_table += '\\end{tabular}\n\\vspace{-1.5em}\n\\end{table*}\n\n\n\n'
    except Exception as e:
        pass

print(my_latex_table)

