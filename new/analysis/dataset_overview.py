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

pointer_dict['airbnb_missing_values'] = {}
pointer_dict['airbnb_missing_values']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/missing_values/dirty_train.csv'
pointer_dict['airbnb_missing_values']['clean_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/missing_values/impute_holoclean_train.csv'
pointer_dict['airbnb_missing_values']['target'] = 'Rating'
pointer_dict['airbnb_missing_values']['drop_labels'] = []

pointer_dict['credit_missing_values'] = {}
pointer_dict['credit_missing_values']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Credit/missing_values/dirty_train.csv'
pointer_dict['credit_missing_values']['clean_path'] = '/home/neutatz/Software/CleanML/data/Credit/missing_values/impute_holoclean_train.csv'
pointer_dict['credit_missing_values']['target'] = "SeriousDlqin2yrs"
pointer_dict['credit_missing_values']['drop_labels'] = []

pointer_dict['marketing_missing_values'] = {}
pointer_dict['marketing_missing_values']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Marketing/missing_values/dirty_train.csv'
pointer_dict['marketing_missing_values']['clean_path'] = '/home/neutatz/Software/CleanML/data/Marketing/missing_values/impute_holoclean_train.csv'
pointer_dict['marketing_missing_values']['target'] = 'Income'
pointer_dict['marketing_missing_values']['drop_labels'] = []

pointer_dict['titanic_missing_values'] = {}
pointer_dict['titanic_missing_values']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Titanic/raw/raw.csv'
pointer_dict['titanic_missing_values']['clean_path'] = '/home/neutatz/Software/CleanML/data/Titanic/raw/Holoclean_mv_clean.csv'
pointer_dict['titanic_missing_values']['target'] = "Survived"
pointer_dict['titanic_missing_values']['drop_labels'] = ['PassengerId', 'Name']

pointer_dict['us_census_missing_values'] = {}
pointer_dict['us_census_missing_values']['dirty_path'] = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/dirty_train.csv'
pointer_dict['us_census_missing_values']['clean_path'] = '/home/neutatz/Software/CleanML/data/USCensus/missing_values/impute_holoclean_train.csv'
pointer_dict['us_census_missing_values']['target'] = "Income"
pointer_dict['us_census_missing_values']['drop_labels'] = []

pointer_dict['company_inconsistency'] = {}
pointer_dict['company_inconsistency']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Company/inconsistency/dirty_train.csv'
pointer_dict['company_inconsistency']['clean_path'] = '/home/neutatz/Software/CleanML/data/Company/inconsistency/clean_train.csv'
pointer_dict['company_inconsistency']['target'] = "Sentiment"
pointer_dict['company_inconsistency']['drop_labels'] = ["Date", "Unnamed: 0", "City"]

pointer_dict['movie_inconsistency'] = {}
pointer_dict['movie_inconsistency']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Movie/inconsistency/dirty_train.csv'
pointer_dict['movie_inconsistency']['clean_path'] = '/home/neutatz/Software/CleanML/data/Movie/inconsistency/clean_train.csv'
pointer_dict['movie_inconsistency']['target'] = "genres"
pointer_dict['movie_inconsistency']['drop_labels'] = []

pointer_dict['restaurant_inconsistency'] = {}
pointer_dict['restaurant_inconsistency']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Restaurant/inconsistency/dirty_train.csv'
pointer_dict['restaurant_inconsistency']['clean_path'] = '/home/neutatz/Software/CleanML/data/Restaurant/inconsistency/clean_train.csv'
pointer_dict['restaurant_inconsistency']['target'] = "priceRange"
pointer_dict['restaurant_inconsistency']['drop_labels'] = ["streetAddress", "telephone", "website"]

pointer_dict['university_inconsistency'] = {}
pointer_dict['university_inconsistency']['dirty_path'] = '/home/neutatz/Software/CleanML/data/University/inconsistency/dirty_train.csv'
pointer_dict['university_inconsistency']['clean_path'] = '/home/neutatz/Software/CleanML/data/University/inconsistency/clean_train.csv'
pointer_dict['university_inconsistency']['target'] = "expenses thous$"
pointer_dict['university_inconsistency']['drop_labels'] = ["university name", "academic-emphasis"]


pointer_dict['airbnb_duplicates'] = {}
pointer_dict['airbnb_duplicates']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/duplicates/dirty_train.csv'
pointer_dict['airbnb_duplicates']['clean_path'] = '/home/neutatz/Software/CleanML/data/Airbnb/duplicates/clean_train.csv'
pointer_dict['airbnb_duplicates']['target'] = 'Rating'
pointer_dict['airbnb_duplicates']['drop_labels'] = []

pointer_dict['citation_duplicates'] = {}
pointer_dict['citation_duplicates']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Citation/duplicates/dirty_train.csv'
pointer_dict['citation_duplicates']['clean_path'] = '/home/neutatz/Software/CleanML/data/Citation/duplicates/clean_train.csv'
pointer_dict['citation_duplicates']['target'] = 'CS'
pointer_dict['citation_duplicates']['drop_labels'] = []

pointer_dict['movie_duplicates'] = {}
pointer_dict['movie_duplicates']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Movie/duplicates/dirty_train.csv'
pointer_dict['movie_duplicates']['clean_path'] = '/home/neutatz/Software/CleanML/data/Movie/duplicates/clean_train.csv'
pointer_dict['movie_duplicates']['target'] = "genres"
pointer_dict['movie_duplicates']['drop_labels'] = []

pointer_dict['restaurant_duplicates'] = {}
pointer_dict['restaurant_duplicates']['dirty_path'] = '/home/neutatz/Software/CleanML/data/Restaurant/duplicates/dirty_train.csv'
pointer_dict['restaurant_duplicates']['clean_path'] = '/home/neutatz/Software/CleanML/data/Restaurant/duplicates/clean_train.csv'
pointer_dict['restaurant_duplicates']['target'] = "priceRange"
pointer_dict['restaurant_duplicates']['drop_labels'] = ["streetAddress", "telephone", "website"]


def get_X_y(data, target_label, drop_labels=[]):
    data_y = data[target_label]
    data_X = data.drop(target_label, 1)
    for drop_label in drop_labels:
        data_X = data_X.drop(drop_label, 1)
    return data_y, data_X

table = ''
for k, v in pointer_dict.items():
    clean = pd.read_csv(v['clean_path'])
    y, X = get_X_y(clean, v['target'], v['drop_labels'])

    number_classes = len(np.unique(y))

    print(str(k) + ': ' + str(X.shape))
    dataset = k.split('_')[0]
    error_type = k.split('_')[-1]
    table += error_type + ' & ' + dataset + ' & ' + str(X.shape[0]) + ' & ' + str(X.shape[1]) + ' & ' + str(number_classes) + '\\\\ \n'

print(table)