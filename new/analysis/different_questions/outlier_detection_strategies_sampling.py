from new.analysis.check_pipelines import dict2table
import copy
import pickle
import numpy as np

import glob
print(glob.glob("/home/neutatz/data/cleanml_results/*_clean.p"))

my_latex_table = ''

def get_hyperparameter_values(mypath):
    results = pickle.load(open(mypath, 'rb'))

    check_components = []
    for i in range(100):
        check_components.append("'data_preprocessing:numerical_transformer:outlier_detection:detection:use_outlier_detection"+ str(i)+"': 'True'",
                        )

    count_classifiers = {}

    count_classifiers['lof'] = []
    count_classifiers['iso'] = []
    count_classifiers['svm'] = []
    count_classifiers['ellipctic'] = []



    for i in range(len(results['models'])):

        count_models_in_ensemble = {}
        count_models_in_ensemble['lof'] = 0
        count_models_in_ensemble['iso'] = 0
        count_models_in_ensemble['svm'] = 0
        count_models_in_ensemble['ellipctic'] = 0
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

            #check if detection is activated
            #count total weight
            total_weight = 0
            low_count = 0
            iso_count = 0
            el_count = 0
            svm_count = 0
            for i in range(100):
                if check_components[i] in str(pipeline):
                    total_weight += 1

                    if "'data_preprocessing:numerical_transformer:outlier_detection:detection:strategy" + str(i) + "': 'lof'" in str(pipeline):
                        low_count += 1
                    if "'data_preprocessing:numerical_transformer:outlier_detection:detection:strategy" + str(i) + "': 'isolation_forest'" in str(pipeline):
                        iso_count += 1
                    if "'data_preprocessing:numerical_transformer:outlier_detection:detection:strategy" + str(i) + "': 'one_class_svm'" in str(pipeline):
                        svm_count += 1
                    if "'data_preprocessing:numerical_transformer:outlier_detection:detection:strategy" + str(i) + "': 'elliptic'" in str(pipeline):
                        el_count += 1

            if total_weight > 0:
                count_models_in_ensemble['lof'] += current_weihgt * (low_count / total_weight)
                count_models_in_ensemble['iso'] += current_weihgt * (iso_count / total_weight)
                count_models_in_ensemble['svm'] += current_weihgt * (svm_count / total_weight)
                count_models_in_ensemble['ellipctic'] += current_weihgt * (el_count / total_weight)

            print(count_models_in_ensemble)

        for k, v in count_models_in_ensemble.items():
            count_classifiers[k].append(v / float(count_models))

    count_components_relative = {}
    for k, v in count_classifiers.items():
        count_components_relative[k] = np.average(v)

    return count_classifiers, check_components, count_components_relative

#_autoclean_1h__dirty.p
for dirty_file in [0]:#glob.glob("/home/neutatz/data/cleanml_results/*autoclean_1h__dirty.p"):
#for dirty_file in glob.glob("/home/neutatz/phd2/cleanML_my/*autoclean_1h__dirty.p"):
#for dirty_file in glob.glob("/home/neutatz/phd2/cleanML_my/*autoclean_1h__dirty.p"):
    dirty_file = "/home/neutatz/data/cleanml_results/marketing_missing_values.py_autoclean__dirty.p"
    task_name = dirty_file.split('/')[-1][:-11]



    count_classifiers_dirty, check_components, count_components_relative_dirty = get_hyperparameter_values(dirty_file)

    my_latex_table += task_name + ' &'
    my_latex_table += '%f, ' % (count_components_relative_dirty['svm'],)
    my_latex_table += '%f, ' % (count_components_relative_dirty['lof'],)
    my_latex_table += '%f, ' % (count_components_relative_dirty['iso'],)
    my_latex_table += '%f, ' % (count_components_relative_dirty['ellipctic'],)
    my_latex_table = my_latex_table[:-2]
    my_latex_table += '\n'

print(my_latex_table)