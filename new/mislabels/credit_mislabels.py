import os
from new.utils import run

if __name__ == "__main__":
    target_label = "SeriousDlqin2yrs"
    drop_labels = []
    clean_path = '/home/neutatz/Software/CleanML/data/Credit/missing_values/impute_holoclean_train.csv'
    save_path = '/home/neutatz/data/cleanml_results/'
    file_name = save_path + os.path.basename(__file__)

    run(clean_path, None, target_label, drop_labels, mislabel_percent=0.05, file_name=file_name)


