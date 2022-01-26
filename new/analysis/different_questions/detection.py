from new.analysis.check_pipelines import dict2table
import copy
import pickle
import numpy as np

import glob
print(glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"))

my_latex_table = ''

def get_hyperparameter_values(mypath):
    results = pickle.load(open(mypath, 'rb'))

    check_components = ["'data_preprocessing:numerical_transformer:outlier_detection:__choice__': 'detection'",
                        "'data_preprocessing:numerical_transformer:outlier_detection:__choice__': 'none'"
                        ]



    count_classifiers = {}

    for check_i in range(len(check_components)):
        count_classifiers[check_components[check_i]] = []



    for i in range(len(results['models'])):

        count_models_in_ensemble = {}
        for check_i in range(len(check_components)):
            count_models_in_ensemble[check_components[check_i]] = 0
        count_models = 0

        print('\n\n\n')
        auto_ml_model = results['models'][i]
        print(auto_ml_model.autosklearn_model.show_models())
        all_models_with_weights = auto_ml_model.autosklearn_model.get_models_with_weights()
        for model_i in range(len(all_models_with_weights)):
            pipeline = auto_ml_model.autosklearn_model.get_models_with_weights()[model_i][1]

            current_weihgt = auto_ml_model.autosklearn_model.get_models_with_weights()[model_i][0]
            print('weight: ' + str(current_weihgt))
            count_models += current_weihgt
            #print(pipeline)
            for check_i in range(len(check_components)):
                if check_components[check_i] in str(pipeline):
                    count_models_in_ensemble[check_components[check_i]] += current_weihgt

        for k, v in count_models_in_ensemble.items():
            count_classifiers[k].append(v / float(count_models))

    count_components_relative = {}
    for k, v in count_classifiers.items():
        count_components_relative[k] = np.average(v)

    return count_classifiers, check_components, count_components_relative

#_autoclean_1h__dirty.p
for dirty_file in glob.glob("/home/neutatz/phd2/cleanML_my/*autoclean__dirty.p"):
#for dirty_file in glob.glob("/home/neutatz/phd2/cleanML_my/*autoclean_1h__dirty.p"):
    task_name = dirty_file.split('/')[-1][:-11]

    try:

        count_classifiers_dirty, check_components, count_components_relative_dirty = get_hyperparameter_values(dirty_file)

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

        my_latex_table += task_name + ' &'
        for check_i in range(len(check_components)):
            my_latex_table += '%f, ' % (count_components_relative_dirty[check_components[check_i]],)
        my_latex_table = my_latex_table[:-2]
        my_latex_table += '\n'
    except Exception as e:
        pass

print(my_latex_table)