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
    args = parser.parse_args()
    file_name = save_path + os.path.basename(__file__) + '_p' + str(args.mislabel_percent) +'_cvfolds_5_'
    run(clean_path, None, target_label, drop_labels, mislabel_percent=args.mislabel_percent, file_name=file_name, avoid_clean_eval=True, cv_folds=5)


