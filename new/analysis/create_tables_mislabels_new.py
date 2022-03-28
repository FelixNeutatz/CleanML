import pickle
import numpy as np

def create_table(files, title):
    my_latex_table = ''

    print(title)
    for file_name in files:
        try:
            p005_val_false = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_p0.01_dirty.p"
            p01_val_false = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_p0.5_dirty.p"
            p005_val_true = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_p0.05_clean_val_True__dirty.p"
            p01_val_true = "/home/neutatz/data/cleanml_results/" + str(file_name) + "_p0.1_clean_val_True__dirty.p"

            my_latex_table += "%s & " % (file_name.split('.')[0],)

            try:
                p005_val_false_f = pickle.load(open(p005_val_false, 'rb'))
                my_latex_table += " $%.2f \\pm %.2f$ &" % (np.average(p005_val_false_f['scores']), np.std(p005_val_false_f['scores']))
            except Exception as e:
                my_latex_table += " NA &"
            try:
                p01_val_false_f = pickle.load(open(p01_val_false, 'rb'))
                my_latex_table += " $%.2f \\pm %.2f$ &" % (np.average(p01_val_false_f['scores']), np.std(p01_val_false_f['scores']))
            except Exception as e:
                my_latex_table += " NA &"
            try:
                p005_val_true_f = pickle.load(open(p005_val_true, 'rb'))
                my_latex_table += " $%.2f \\pm %.2f$ &" % (np.average(p005_val_true_f['scores']), np.std(p005_val_true_f['scores']))
            except Exception as e:
                my_latex_table += " NA &"
            try:
                p01_val_true_f = pickle.load(open(p01_val_true, 'rb'))
                my_latex_table += " $%.2f \\pm %.2f$ \\\\ \n" % (np.average(p01_val_true_f['scores']), np.std(p01_val_true_f['scores']))
            except Exception as e:
                my_latex_table += " NA  \\\\ \n"
        except Exception as e:
            pass

    print(my_latex_table)
    print('\n\n')

create_table(['credit_mislabels.py', 'eeg_mislabels.py', 'us_census_mislabels.py'], 'mislabels')
