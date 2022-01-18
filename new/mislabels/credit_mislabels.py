import os
from new.utils import run
import argparse

if __name__ == "__main__":
    target_label = "SeriousDlqin2yrs"
    drop_labels = []
    clean_path = '/home/neutatz/Software/CleanML/data/Credit/missing_values/impute_holoclean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-mislabel_percent', type=float)
    parser.add_argument('-clean_validation_labels')
    args = parser.parse_args()
    clean_validation_labels = args.clean_validation_labels == 'True'
    file_name = save_path + os.path.basename(__file__) + '_p' + str(args.mislabel_percent) +'_clean_val_' + str(clean_validation_labels) + '_'
    run(clean_path, None, target_label, drop_labels, mislabel_percent=args.mislabel_percent, file_name=file_name, clean_validation_labels=clean_validation_labels, avoid_clean_eval=True)


