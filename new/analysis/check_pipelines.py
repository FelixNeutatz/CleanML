import pickle
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path')
args = parser.parse_args()

print(args.path)

results = pickle.load(open(args.path, 'rb'))

print(results['scores'])

check_classifiers = ["'classifier:__choice__': 'adaboost'",
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
 "'classifier:__choice__': 'sgd'"]
check_feature_processors = ["'feature_preprocessor:__choice__': 'densifier'",
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
 "'feature_preprocessor:__choice__': 'truncatedSVD'"]

count_classifiers = {}
count_feature_processors = {}

for check_i in range(len(check_classifiers)):
    count_classifiers[check_classifiers[check_i]] = 0

for check_i in range(len(check_feature_processors)):
    count_feature_processors[check_feature_processors[check_i]] = 0

print(count_feature_processors)

count_models = 0

for i in range(len(results['models'])):
    print('\n\n\n')
    auto_ml_model = results['models'][i]
    #print(auto_ml_model.autosklearn_model.show_models())
    all_models_with_weights = auto_ml_model.autosklearn_model.get_models_with_weights()
    for model_i in range(len(all_models_with_weights)):
        pipeline = auto_ml_model.autosklearn_model.get_models_with_weights()[model_i][1]
        count_models += 1
        print(pipeline)
        for check_i in range(len(check_classifiers)):
            if check_classifiers[check_i] in str(pipeline):
                count_classifiers[check_classifiers[check_i]] += 1
        for check_i in range(len(check_feature_processors)):
            if check_feature_processors[check_i] in str(pipeline):
                count_feature_processors[check_feature_processors[check_i]] += 1

print(count_classifiers)
print(np.sum(list(count_classifiers.values())))
print(len(auto_ml_model.autosklearn_model.get_models_with_weights()))


print(count_feature_processors)
print(np.sum(list(count_feature_processors.values())))

print(count_models)

for i in range(len(results['feature_importances'])):
    print('\n\n\n')
    feature_importances = results['feature_importances'][i]
    print(feature_importances)