from new.analysis.check_pipelines import get_hyperparameter_values
from new.analysis.check_pipelines import dict2table
import copy

import glob
print(glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"))

my_latex_table = ''

for clean_file in glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"):
    task_name = clean_file.split('/')[-1][:-11]
    dirty_file = clean_file[:-7] + 'dirty.p'

    try:

        count_classifiers_clean, check_components, count_components_relative_clean, count_models_clean = get_hyperparameter_values(clean_file)
        count_classifiers_dirty, _, count_components_relative_dirty, count_models_dirty = get_hyperparameter_values(dirty_file)

        if len(my_latex_table) == 0:
            my_latex_table = '\\toprule\n'
            my_latex_table += 'Task &'
            for check_i in range(len(check_components)):
                my_latex_table += check_components[check_i] + ' & '
            my_latex_table = my_latex_table[:-2]
            my_latex_table += '\\\\ \n'
            my_latex_table += '\\midrule\n'

        dict2table(check_components, count_classifiers_dirty)
        #dict2table(check_components, count_classifiers_clean)

        count_diff_rel = {}
        for k, v in count_components_relative_clean.items():
            count_diff_rel[k] = v - count_components_relative_dirty[k]

        my_latex_table += task_name + ' &'
        for check_i in range(len(check_components)):
            my_latex_table += str(count_diff_rel[check_components[check_i]]) + ' & '
        my_latex_table = my_latex_table[:-2]
        my_latex_table += '\\\\ \n'
    except Exception as e:
        pass

print(my_latex_table)