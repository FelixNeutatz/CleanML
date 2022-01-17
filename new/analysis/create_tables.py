import pickle
import numpy as np

my_latex_table = ''

for file_name in ['airbnb_outliers.py', 'credit_outliers.py', 'eeg_outliers.py', 'sensor_outliers.py']:
    my_path_clean = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_clean.p"
    my_path_dirty = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_dirty.p"

    result_clean = pickle.load(open(my_path_clean, 'rb'))
    result_dirty = pickle.load(open(my_path_dirty, 'rb'))

    my_latex_table += "%s & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\ \n" % (file_name.split('.')[0], np.average(result_dirty['scores']), np.std(result_dirty['scores']), np.average(result_clean['scores']), np.std(result_clean['scores']))

print(my_latex_table)