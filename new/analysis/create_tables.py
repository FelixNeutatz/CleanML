import pickle
import numpy as np

def create_table(files, title):
    my_latex_table = ''

    print(title)
    for file_name in files:
        my_path_clean = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_clean.p"
        my_path_dirty = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_dirty.p"

        result_clean = pickle.load(open(my_path_clean, 'rb'))
        result_dirty = pickle.load(open(my_path_dirty, 'rb'))

        my_latex_table += "%s & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\ \n" % (file_name.split('.')[0], np.average(result_dirty['scores']), np.std(result_dirty['scores']), np.average(result_clean['scores']), np.std(result_clean['scores']))

    print(my_latex_table)
    print('\n\n')

create_table(['airbnb_outliers.py', 'credit_outliers.py', 'eeg_outliers.py', 'sensor_outliers.py'], 'outliers')
create_table(['airbnb_missing_values.py','credit_missing_values.py','marketing_missing_values.py','titanic_missing_values.py','us_census_missing_values.py'], 'missing values')
create_table(['company_inconsistency.py', 'movie_inconsistency.py', 'restaurant_inconsistency.py', 'university_inconsistency.py'], 'inconsistency')
create_table(['airbnb_duplicates.py','citation_duplicates.py','movie_duplicates.py','restaurant_duplicates.py'], 'duplicates')