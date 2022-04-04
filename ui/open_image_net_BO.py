from skimage.io import imread, imshow
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from glob import glob
import optuna
from optuna.integration import TFKerasPruningCallback

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Objective(object):
    def __init__(self):
        # Hold this implementation specific arguments as the fields of the class.
        pass

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.

        batch_size = trial.suggest_int("batch_size", 8, 100, log=True)
        #batch_size = 8

        train = tf.keras.preprocessing.image_dataset_from_directory(
            '/home/neutatz/phd2/new_image_test/train',
            labels="inferred",
            label_mode="categorical",
            class_names=my_classes,
            shuffle=True,
            seed=123,
            batch_size=batch_size,
            image_size=(32, 32),
        )

        valid = tf.keras.preprocessing.image_dataset_from_directory(
            '/home/neutatz/phd2/new_image_test/val',
            labels="inferred",
            label_mode="categorical",
            class_names=my_classes,
            shuffle=False,
            seed=123,
            batch_size=batch_size,
            image_size=(32, 32),
        )

        test = tf.keras.preprocessing.image_dataset_from_directory(
            '/home/neutatz/phd2/new_image_test/test',
            labels="inferred",
            label_mode="categorical",
            class_names=my_classes,
            shuffle=False,
            seed=123,
            batch_size=len(test_index),
            image_size=(32, 32),
        )

        use_imagenet = trial.suggest_categorical("use_imagenet", [True, False])
        weights = None
        if use_imagenet:
            weights = 'imagenet'

        base_model = tf.keras.applications.ResNet50(
            input_shape=(32, 32, 3),
            include_top=False,
            weights=weights,
        )


        which_layer = trial.suggest_categorical("layer", ["conv2_block1_out","conv2_block2_out","conv2_block3_out","conv3_block1_out","conv3_block2_out","conv3_block3_out","conv3_block4_out","conv4_block1_out","conv4_block2_out","conv4_block3_out","conv4_block4_out","conv4_block5_out","conv4_block6_out","conv5_block1_out","conv5_block2_out","conv5_block3_out"])

        base_model = tf.keras.Model(
            base_model.inputs,
            outputs=[base_model.get_layer(which_layer).output]
        )

        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.applications.resnet.preprocess_input(inputs)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(10)(x)
        model = tf.keras.Model(inputs, x)

        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.summary()
        loss_0, acc_0 = model.evaluate(valid)
        print(f"loss {loss_0}, acc {acc_0}")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "best_model",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )

        epochs = trial.suggest_int("epochs", 10, 100, log=True)
        #epochs = 100

        history = model.fit(
            train,
            validation_data=valid,
            epochs=epochs,  # 100
            callbacks=[checkpoint,
                       #tf.keras.callbacks.EarlyStopping(patience=3)
                       TFKerasPruningCallback(trial, 'val_accuracy')],
        )

        model.load_weights("best_model")

        loss, acc = model.evaluate(valid)
        print(f"final loss {loss}, final acc {acc}")

        test_loss, test_acc = model.evaluate(test)
        print(f"test loss {test_loss}, test acc {test_acc}")

        predictions = model.predict(test)

        y_classes = predictions.argmax(axis=-1)
        y_pred_classes = []
        for ii in y_classes:
            y_pred_classes.append(ii)

        y_true = []
        for next_element in test:
            for ii in range(len(next_element[1])):
                for cii in range(10):
                    if float(next_element[1][ii][cii]) == 1.0:
                        y_true.append(cii)

        trial.set_user_attr('test_balanced_accuracy', balanced_accuracy_score(y_true, y_pred_classes))

        return acc



def get_score(X, y, train_index, test_index):
    skf1 = StratifiedKFold(n_splits=3)
    fold_ids1 = list(skf1.split(X[train_index, :], y[train_index]))
    train_index1, valid_index1 = fold_ids1[0]
    training_list = []
    valid_list = []
    for ii in train_index1:
        training_list.append(train_index[ii])
    for ii in valid_index1:
        valid_list.append(train_index[ii])

    filelist = glob("/home/neutatz/phd2/new_image_test/*/*/*.png")
    for f in filelist:
        os.remove(f)

    for ii in training_list:
        shutil.copyfile(files_all[ii],
                        ('/home/neutatz/phd2/new_image_test/train/' + str(y[ii]) + '/image' + str(ii) + '.png'))

    for ii in valid_list:
        shutil.copyfile(files_all[ii],
                        ('/home/neutatz/phd2/new_image_test/val/' + str(y[ii]) + '/image' + str(ii) + '.png'))

    for ii in test_index:
        shutil.copyfile(files_all[ii],
                        ('/home/neutatz/phd2/new_image_test/test/' + str(y[ii]) + '/image' + str(ii) + '.png'))

    try_these_parameters_first = [{'batch_size': 8, 'use_imagenet': False, 'layer': "conv2_block3_out", "learning_rate": 0.0001, "epochs": 100},
                                  {'batch_size': 20, 'use_imagenet': False, 'layer': "conv2_block3_out", "learning_rate": 0.0001, "epochs": 10}]

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
    study.enqueue_trial(try_these_parameters_first[0])
    study.optimize(Objective(), timeout=60*60)

    result = study.best_trial.user_attrs['test_balanced_accuracy']
    print('Result: ' + str(result))
    print('Best Params: ' + str(study.best_params))
    return result


if __name__ == '__main__':


    my_classes = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

    tf.random.set_seed(123)

    y = []
    X = []
    indices = []
    files_all = []

    for file in glob("/home/neutatz/phd2/data_centric/data32/*/*/*.png"):
        y.append(file.split('/')[-2])
        indices.append(file.split('/')[-1])
        files_all.append(file)

        image = imread(file)
        feature_vector = np.reshape(image, (32 * 32 * 3))
        X.append(image)

    y_clean = []
    X_clean = []
    indices_clean = []
    files_all_clean = []

    for file in glob("/home/neutatz/phd2/data_centric/new_data32_clean/*/*.png"):
        y_clean.append(file.split('/')[-2])
        indices_clean.append(file.split('/')[-1])
        files_all_clean.append(file)

        image = imread(file)
        feature_vector = np.reshape(image, (32 * 32 * 3))
        X_clean.append(image)

    y = np.array(y)
    X = np.array(X)

    print(X)
    print(y)

    skf = StratifiedKFold(n_splits=5)
    fold_ids = list(skf.split(X, y))



    scores_clean = []
    scores_dirty = []

    for train_index, test_index in fold_ids:

        test_index_new = []
        for iii in test_index:
            if indices[iii] in indices_clean:
                test_index_new.append(iii)
        test_index = test_index_new

        train_clean_index = []
        for iii in train_index:
            if indices[iii] in indices_clean:
                train_clean_index.append(iii)

        s_dirty = get_score(X, y, train_index, test_index)
        print('BO dirty: ' + str(s_dirty))
        s_clean = get_score(X, y, train_clean_index, test_index)

        print('BO dirty: ' + str(s_dirty))
        print('BO clean: ' + str(s_clean))

        scores_dirty.append(s_dirty)
        scores_clean.append(s_clean)

    print('BO dirty: ' + str(np.average(scores_dirty)) + ' +- ' + str(np.std(scores_dirty)))
    print('BO clean: ' + str(np.average(scores_clean)) + ' +- ' + str(np.std(scores_clean)))
