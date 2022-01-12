import pickle

results = pickle.load(open('/tmp/test.p', 'rb'))

print(results['scores'])

for i in range(len(results['models'])):
    print('\n\n\n')
    auto_ml_model = results['models'][i]
    print(auto_ml_model.autosklearn_model.show_models())

for i in range(len(results['feature_importances'])):
    print('\n\n\n')
    feature_importances = results['feature_importances'][i]
    print(feature_importances)