import pickle
import numpy as np
import pandas as pd
import copy

import argparse

def dict2table(check_components, count_classifiers):
    table = ''
    for check_i in range(len(check_components)):
        table += check_components[check_i] + ' & '
    table += '\n'
    for check_i in range(len(check_components)):
        table += str(count_classifiers[check_components[check_i]]) + ' & '
    print(table + '\n\n')

def get_hyperparameter_values(mypath):
    results = pickle.load(open(mypath, 'rb'))

    check_components = ["'classifier:__choice__': 'adaboost'",
                        "'classifier:__choice__': 'bernoulli_nb'",
                        "'classifier:__choice__': 'decision_tree'",
                        "'classifier:__choice__': 'extra_trees'",
                        "'classifier:__choice__': 'gaussian_nb'",
                        "'classifier:__choice__': 'gradient_boosting'",
                        "'classifier:__choice__': 'k_nearest_neighbors'",
                        "'classifier:__choice__': 'lda'",
                        "'classifier:__choice__': 'liblinear_svc'",
                        "'classifier:__choice__': 'libsvm_svc'",
                        "'classifier:__choice__': 'mlp'",
                        "'classifier:__choice__': 'multinomial_nb'",
                        "'classifier:__choice__': 'passive_aggressive'",
                        "'classifier:__choice__': 'qda'",
                        "'classifier:__choice__': 'random_forest'",
                        "'classifier:__choice__': 'sgd'",

                        "'feature_preprocessor:__choice__': 'densifier'",
                        "'feature_preprocessor:__choice__': 'extra_trees_preproc_for_classification'",
                        "'feature_preprocessor:__choice__': 'fast_ica'",
                        "'feature_preprocessor:__choice__': 'feature_agglomeration'",
                        "'feature_preprocessor:__choice__': 'kernel_pca'",
                        "'feature_preprocessor:__choice__': 'kitchen_sinks'",
                        "'feature_preprocessor:__choice__': 'liblinear_svc_preprocessor'",
                        "'feature_preprocessor:__choice__': 'no_preprocessing'",
                        "'feature_preprocessor:__choice__': 'nystroem_sampler'",
                        "'feature_preprocessor:__choice__': 'pca'",
                        "'feature_preprocessor:__choice__': 'polynomial'",
                        "'feature_preprocessor:__choice__': 'random_trees_embedding'",
                        "'feature_preprocessor:__choice__': 'select_rates_classification'",
                        "'feature_preprocessor:__choice__': 'truncatedSVD'",

                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'quantile_transformer'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'minmax'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'none'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'normalize'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'power_transformer'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'quantile_transformer'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'robust_scaler'",
                        "'data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__': 'standardize'",

                        "'data_preprocessor:feature_type:categorical_transformer:categorical_encoding:__choice__': 'encoding'",
                        "'data_preprocessor:feature_type:categorical_transformer:categorical_encoding:__choice__': 'no_encoding'",
                        "'data_preprocessor:feature_type:categorical_transformer:categorical_encoding:__choice__': 'one_hot_encoding'",

                        "'data_preprocessor:feature_type:numerical_transformer:imputation:strategy': 'most_frequent'",
                        "'data_preprocessor:feature_type:numerical_transformer:imputation:strategy': 'mean'",
                        "'data_preprocessor:feature_type:numerical_transformer:imputation:strategy': 'median'",

                        "'data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__': 'no_coalescense'",
                        "'data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__': 'minority_coalescer'",

                        "'balancing:strategy': 'weighting'",
                        "'balancing:strategy': 'none'",

                        ]

    count_classifiers = {}

    for check_i in range(len(check_components)):
        count_classifiers[check_components[check_i]] = 0

    count_models = 0

    for i in range(len(results['models'])):
        print('\n\n\n')
        auto_ml_model = results['models'][i]
        # print(auto_ml_model.autosklearn_model.show_models())
        all_models_with_weights = auto_ml_model.autosklearn_model.get_models_with_weights()
        for model_i in range(len(all_models_with_weights)):
            pipeline = auto_ml_model.autosklearn_model.get_models_with_weights()[model_i][1]
            count_models += 1
            #print(pipeline)
            for check_i in range(len(check_components)):
                if check_components[check_i] in str(pipeline):
                    count_classifiers[check_components[check_i]] += 1

    #print(count_classifiers)
    #print(np.sum(list(count_classifiers.values())))
    #print(len(auto_ml_model.autosklearn_model.get_models_with_weights()))

    count_components_relative = copy.deepcopy(count_classifiers)
    for k,v in count_components_relative.items():
        count_components_relative[k] = v / float(count_models)

    return count_classifiers, check_components, count_components_relative, count_models

''' 
    for i in range(len(results['feature_importances'])):
    print('\n\n\n')
    feature_importances = results['feature_importances'][i]
    print(feature_importances)
'''